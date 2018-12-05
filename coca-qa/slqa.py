import logging
import torch
import numpy as np
from torch.nn.functional import cross_entropy, nll_loss
from typing import Optional, Dict, List, Any

from allennlp.models.model import Model
from allennlp.data import Vocabulary
from allennlp.modules import TextFieldEmbedder, Seq2SeqEncoder
from allennlp.nn import RegularizerApplicator
from allennlp.training.metrics import CategoricalAccuracy, BooleanAccuracy, SquadEmAndF1, Average
from allennlp.nn import InitializerApplicator, util
from allennlp.modules.input_variational_dropout import InputVariationalDropout

from allennlp.tools import squad_eval

from models.layers import FusionLayer, BilinearSeqAtt

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


@Model.register("slqa")
class MultiGranularityHierarchicalAttentionFusionNetworks(Model):

    def __init__(self, vocab: Vocabulary,
                 text_field_embedder: TextFieldEmbedder,
                 phrase_layer: Seq2SeqEncoder,
                 projected_layer: Seq2SeqEncoder,
                 contextual_passage: Seq2SeqEncoder,
                 contextual_question: Seq2SeqEncoder,
                 dropout: float = 0.2,
                 regularizer: Optional[RegularizerApplicator] = None,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 ):

        super(MultiGranularityHierarchicalAttentionFusionNetworks, self).__init__(vocab, regularizer)
        self._text_field_embedder = text_field_embedder
        self._phrase_layer = phrase_layer
        self._encoding_dim = self._phrase_layer.get_output_dim()
        self.projected_layer = torch.nn.Linear(self._encoding_dim + 1024, self._encoding_dim)
        self.fuse = FusionLayer(self._encoding_dim)
        self.projected_lstm = projected_layer
        self.contextual_layer_p = contextual_passage
        self.contextual_layer_q = contextual_question
        self.linear_self_align = torch.nn.Linear(self._encoding_dim, 1)
        self.bilinear_layer_s = BilinearSeqAtt(self._encoding_dim, self._encoding_dim)
        self.bilinear_layer_e = BilinearSeqAtt(self._encoding_dim, self._encoding_dim)
        self.yesno_predictor = torch.nn.Linear(self._encoding_dim, 3)
        self.relu = torch.nn.ReLU()

        self._max_span_length = 30

        self._span_start_accuracy = CategoricalAccuracy()
        self._span_end_accuracy = CategoricalAccuracy()
        self._span_accuracy = BooleanAccuracy()
        self._squad_metrics = SquadEmAndF1()
        self._span_yesno_accuracy = CategoricalAccuracy()
        self._official_f1 = Average()
        self._variational_dropout = InputVariationalDropout(dropout)

        self._loss = torch.nn.CrossEntropyLoss()
        initializer(self)

    def forward(self, question: Dict[str, torch.LongTensor],
                passage: Dict[str, torch.LongTensor],
                span_start: torch.IntTensor = None,
                span_end: torch.IntTensor = None,
                yesno_list: torch.IntTensor = None,
                metadata: List[Dict[str, Any]] = None) -> Dict[str, torch.Tensor]:

        batch_size, max_qa_count, max_q_len, _ = question['token_characters'].size()
        total_qa_count = batch_size * max_qa_count
        qa_mask = torch.ge(yesno_list, 0).view(total_qa_count)

        embedded_question = self._text_field_embedder(question, num_wrapping_dims=1)
        # total_qa_count * max_q_len * encoding_dim
        embedded_question = embedded_question.reshape(total_qa_count, max_q_len,
                                                      self._text_field_embedder.get_output_dim())
        embedded_passage = self._text_field_embedder(passage)

        word_emb_ques, elmo_ques, ques_feat = torch.split(embedded_question, [200, 1024, 40], dim=2)
        word_emb_pass, elmo_pass, pass_feat = torch.split(embedded_passage, [200, 1024, 40], dim=2)
        embedded_question = torch.cat([word_emb_ques, elmo_ques], dim=2)
        embedded_passage = torch.cat([word_emb_pass, elmo_pass], dim=2)

        embedded_question = self._variational_dropout(embedded_question)
        embedded_passage = self._variational_dropout(embedded_passage)
        passage_length = embedded_passage.size(1)

        question_mask = util.get_text_field_mask(question, num_wrapping_dims=1).float()
        question_mask = question_mask.reshape(total_qa_count, max_q_len)
        passage_mask = util.get_text_field_mask(passage).float()

        repeated_passage_mask = passage_mask.unsqueeze(1).repeat(1, max_qa_count, 1)
        repeated_passage_mask = repeated_passage_mask.view(total_qa_count, passage_length)

        encode_passage = self._phrase_layer(embedded_passage, passage_mask)
        projected_passage = self.relu(self.projected_layer(torch.cat([encode_passage, elmo_pass], dim=2)))

        encode_question = self._phrase_layer(embedded_question, question_mask)
        projected_question = self.relu(self.projected_layer(torch.cat([encode_question, elmo_ques], dim=2)))

        encoded_passage = self._variational_dropout(projected_passage)
        repeated_encoded_passage = encoded_passage.unsqueeze(1).repeat(1, max_qa_count, 1, 1)
        repeated_encoded_passage = repeated_encoded_passage.view(total_qa_count,
                                                                 passage_length,
                                                                 self._encoding_dim)
        repeated_pass_feat = (pass_feat.unsqueeze(1).repeat(1, max_qa_count, 1, 1)).view(total_qa_count,
                                                                                         passage_length,
                                                                                         40)
        encoded_question = self._variational_dropout(projected_question)

        # total_qa_count * max_q_len * passage_length
        # cnt * m * n
        s = torch.bmm(encoded_question, repeated_encoded_passage.transpose(2, 1))
        alpha = util.masked_softmax(s, question_mask.unsqueeze(2).expand(s.size()), dim=1)
        # cnt * n * h
        aligned_p = torch.bmm(alpha.transpose(2, 1), encoded_question)

        # cnt * m * n
        beta = util.masked_softmax(s, repeated_passage_mask.unsqueeze(1).expand(s.size()), dim=2)
        # cnt * m * h
        aligned_q = torch.bmm(beta, repeated_encoded_passage)

        fused_p = self.fuse(repeated_encoded_passage, aligned_p)
        fused_q = self.fuse(encoded_question, aligned_q)

        # add manual features here
        q_aware_p = self.projected_lstm(torch.cat([fused_p, repeated_pass_feat], dim=2), repeated_passage_mask)

        # cnt * n * n
        self_p = torch.bmm(q_aware_p, q_aware_p.transpose(2, 1))
        for i in range(passage_length):
            self_p[:, i, i] = 0
        lamb = util.masked_softmax(self_p, repeated_passage_mask.unsqueeze(1).expand(self_p.size()), dim=2)
        # cnt * n * h
        self_aligned_p = torch.bmm(lamb, q_aware_p)

        # cnt * n * h
        fused_self_p = self.fuse(q_aware_p, self_aligned_p)
        contextual_p = self.contextual_layer_p(fused_self_p, repeated_passage_mask)

        contextual_q = self.contextual_layer_q(fused_q, question_mask)
        # cnt * m
        gamma = util.masked_softmax(self.linear_self_align(contextual_q).squeeze(2), question_mask, dim=1)
        # cnt * h
        weighted_q = torch.bmm(gamma.unsqueeze(1), contextual_q).squeeze(1)

        span_start_logits = self.bilinear_layer_s(weighted_q, contextual_p)
        span_end_logits = self.bilinear_layer_e(weighted_q, contextual_p)

        # cnt * n * 1  cnt * 1 * h
        span_yesno_logits = self.yesno_predictor(torch.bmm(span_end_logits.unsqueeze(2), weighted_q.unsqueeze(1)))

        span_start_logits = util.replace_masked_values(span_start_logits, repeated_passage_mask, -1e7)
        span_end_logits = util.replace_masked_values(span_end_logits, repeated_passage_mask, -1e7)

        best_span = self._get_best_span_yesno_followup(span_start_logits, span_end_logits,
                                                       span_yesno_logits, self._max_span_length)

        output_dict: Dict[str, Any] = {}

        # Compute the loss for training

        if span_start is not None:
            loss = nll_loss(util.masked_log_softmax(span_start_logits, repeated_passage_mask), span_start.view(-1),
                            ignore_index=-1)
            self._span_start_accuracy(span_start_logits, span_start.view(-1), mask=qa_mask)
            loss += nll_loss(util.masked_log_softmax(span_end_logits,
                                                     repeated_passage_mask), span_end.view(-1), ignore_index=-1)
            self._span_end_accuracy(span_end_logits, span_end.view(-1), mask=qa_mask)
            self._span_accuracy(best_span[:, 0:2],
                                torch.stack([span_start, span_end], -1).view(total_qa_count, 2),
                                mask=qa_mask.unsqueeze(1).expand(-1, 2).long())
            # add a select for the right span to compute loss
            gold_span_end_loc = []
            span_end = span_end.view(total_qa_count).squeeze().data.cpu().numpy()
            for i in range(0, total_qa_count):
                gold_span_end_loc.append(max(span_end[i] * 3 + i * passage_length * 3, 0))
                gold_span_end_loc.append(max(span_end[i] * 3 + i * passage_length * 3 + 1, 0))
                gold_span_end_loc.append(max(span_end[i] * 3 + i * passage_length * 3 + 2, 0))
            gold_span_end_loc = span_start.new(gold_span_end_loc)
            pred_span_end_loc = []
            for i in range(0, total_qa_count):
                pred_span_end_loc.append(max(best_span[i][1] * 3 + i * passage_length * 3, 0))
                pred_span_end_loc.append(max(best_span[i][1] * 3 + i * passage_length * 3 + 1, 0))
                pred_span_end_loc.append(max(best_span[i][1] * 3 + i * passage_length * 3 + 2, 0))
            predicted_end = span_start.new(pred_span_end_loc)

            _yesno = span_yesno_logits.view(-1).index_select(0, gold_span_end_loc).view(-1, 3)
            loss += nll_loss(torch.nn.functional.log_softmax(_yesno, dim=-1), yesno_list.view(-1), ignore_index=-1)

            _yesno = span_yesno_logits.view(-1).index_select(0, predicted_end).view(-1, 3)
            self._span_yesno_accuracy(_yesno, yesno_list.view(-1), mask=qa_mask)

            output_dict["loss"] = loss

        # Compute the EM and F1 on SQuAD and add the tokenized input to the output.
        output_dict['best_span_str'] = []
        output_dict['qid'] = []
        output_dict['yesno'] = []
        best_span_cpu = best_span.detach().cpu().numpy()
        for i in range(batch_size):
            passage_str = metadata[i]['original_passage']
            offsets = metadata[i]['token_offsets']
            f1_score = 0.0
            per_dialog_best_span_list = []
            per_dialog_yesno_list = []
            per_dialog_query_id_list = []
            for per_dialog_query_index, (iid, answer_texts) in enumerate(
                    zip(metadata[i]["instance_id"], metadata[i]["answer_texts_list"])):
                predicted_span = tuple(best_span_cpu[i * max_qa_count + per_dialog_query_index])
                start_offset = offsets[predicted_span[0]][0]
                end_offset = offsets[predicted_span[1]][1]
                yesno_pred = predicted_span[2]
                per_dialog_yesno_list.append(yesno_pred)
                per_dialog_query_id_list.append(iid)
                best_span_string = passage_str[start_offset:end_offset]
                per_dialog_best_span_list.append(best_span_string)
                if answer_texts:
                    if len(answer_texts) > 1:
                        t_f1 = []
                        # Compute F1 over N-1 human references and averages the scores.
                        for answer_index in range(len(answer_texts)):
                            idxes = list(range(len(answer_texts)))
                            idxes.pop(answer_index)
                            refs = [answer_texts[z] for z in idxes]
                            t_f1.append(squad_eval.metric_max_over_ground_truths(squad_eval.f1_score,
                                                                                 best_span_string,
                                                                                 refs))
                        f1_score = 1.0 * sum(t_f1) / len(t_f1)
                    else:
                        f1_score = squad_eval.metric_max_over_ground_truths(squad_eval.f1_score,
                                                                            best_span_string,
                                                                            answer_texts)
                self._official_f1(100 * f1_score)
            output_dict['qid'].append(per_dialog_query_id_list)
            output_dict['best_span_str'].append(per_dialog_best_span_list)
            output_dict['yesno'].append(per_dialog_yesno_list)
        return output_dict

    def decode(self, output_dict: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        yesno_tags = [[self.vocab.get_token_from_index(x, namespace="yesno_labels") for x in yn_list] \
                      for yn_list in output_dict.pop("yesno")]
        output_dict['yesno'] = yesno_tags
        return output_dict

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {'start_acc': self._span_start_accuracy.get_metric(reset),
                'end_acc': self._span_end_accuracy.get_metric(reset),
                'span_acc': self._span_accuracy.get_metric(reset),
                'yesno': self._span_yesno_accuracy.get_metric(reset),
                'f1': self._official_f1.get_metric(reset), }

    @staticmethod
    def _get_best_span_yesno_followup(span_start_logits: torch.Tensor,
                                      span_end_logits: torch.Tensor,
                                      span_yesno_logits: torch.Tensor,
                                      max_span_length: int) -> torch.Tensor:
        if span_start_logits.dim() != 2 or span_end_logits.dim() != 2:
            raise ValueError("Input shapes must be (batch_size, passage_length)")
        batch_size, passage_length = span_start_logits.size()
        max_span_log_prob = [-1e20] * batch_size
        span_start_argmax = [0] * batch_size
        best_word_span = span_start_logits.new_zeros((batch_size, 3), dtype=torch.long)
        span_start_logits = span_start_logits.data.cpu().numpy()
        span_end_logits = span_end_logits.data.cpu().numpy()
        span_yesno_logits = span_yesno_logits.data.cpu().numpy()

        for b_i in range(batch_size):  # pylint: disable=invalid-name
            for j in range(passage_length):
                val1 = span_start_logits[b_i, span_start_argmax[b_i]]
                if val1 < span_start_logits[b_i, j]:
                    span_start_argmax[b_i] = j
                    val1 = span_start_logits[b_i, j]
                val2 = span_end_logits[b_i, j]
                if val1 + val2 > max_span_log_prob[b_i]:
                    if j - span_start_argmax[b_i] > max_span_length:
                        continue
                    best_word_span[b_i, 0] = span_start_argmax[b_i]
                    best_word_span[b_i, 1] = j
                    max_span_log_prob[b_i] = val1 + val2
        for b_i in range(batch_size):
            j = best_word_span[b_i, 1]
            yesno_pred = np.argmax(span_yesno_logits[b_i, j])
            best_word_span[b_i, 2] = int(yesno_pred)
        return best_word_span
