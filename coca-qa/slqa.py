import logging
import torch
from torch.nn import LSTM
from torch.nn.parameter import Parameter
from torch.nn.functional import cross_entropy, nll_loss
from typing import Optional, Dict, List, Any

from allennlp.models.model import Model
from allennlp.data import Vocabulary
from allennlp.modules import TextFieldEmbedder, Seq2SeqEncoder, TimeDistributed
from allennlp.nn import RegularizerApplicator, InitializerApplicator
from allennlp.training.metrics import CategoricalAccuracy, BooleanAccuracy, SquadEmAndF1
from allennlp.nn import util
from models.fusion_layer import FusionLayer
from models.self_attention_layer import SelfAttentionLayer
from utils.vector_weight_sum import attention_weight_sum_batch

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


@Model.register("slqa")
class MultiGranularityHierarchicalAttentionFusionNetworks(Model):

    def __init__(self, vocab: Vocabulary,
                 text_field_embedder: TextFieldEmbedder,
                 phrase_layer: Seq2SeqEncoder,
                 passage_self_attention: Seq2SeqEncoder,
                 semantic_rep_layer: Seq2SeqEncoder,
                 contextual_question_layer: Seq2SeqEncoder,
                 dropout: float = 0.2,
                 mask_lstms: bool = True,
                 regularizer: Optional[RegularizerApplicator] = None,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 ):

        super(MultiGranularityHierarchicalAttentionFusionNetworks, self).__init__(vocab, regularizer)
        self._text_field_embedder = text_field_embedder
        self._phrase_layer = phrase_layer
        # self._highway_layer = TimeDistributed(Highway(text_field_embedder.get_output_dim(),
        #                                               num_highway_layers))
        self._encoding_dim = self._phrase_layer.get_output_dim()
        # self._atten_linear_layer = TimeDistributed(torch.nn.Linear(in_features=self._encoding_dim,
        #                                                            out_features=self._encoding_dim, bias=False))
        self._atten_linear_layer = torch.nn.Linear(in_features=self._encoding_dim,
                                                   out_features=self._encoding_dim, bias=False)
        self._relu = torch.nn.ReLU()
        self._softmax_d1 = torch.nn.Softmax(dim=1)
        self._softmax_d2 = torch.nn.Softmax(dim=2)

        self._atten_fusion = FusionLayer(self._encoding_dim)

        self._tanh = torch.nn.Tanh()
        self._sigmoid = torch.nn.Sigmoid()

        self._passage_self_attention = passage_self_attention

        # self._self_atten_layer = SelfAttentionLayer(self._encoding_dim)
        self._self_atten_layer = torch.nn.Bilinear(self._encoding_dim, self._encoding_dim, self._encoding_dim,
                                                   bias=False)
        self._self_atten_fusion = FusionLayer(self._encoding_dim)

        self._semantic_rep_layer = semantic_rep_layer
        self._contextual_question_layer = contextual_question_layer

        # self._vector_linear = TimeDistributed(
        #     torch.nn.Linear(in_features=self._encoding_dim, out_features=1, bias=False))
        #
        # self._model_layer_s = TimeDistributed(
        #     torch.nn.Linear(in_features=self._encoding_dim, out_features=self._encoding_dim, bias=False))
        # self._model_layer_e = TimeDistributed(
        #     torch.nn.Linear(in_features=self._encoding_dim, out_features=self._encoding_dim, bias=False))
        self._vector_linear = torch.nn.Linear(in_features=self._encoding_dim, out_features=1, bias=False)

        self._model_layer_s = torch.nn.Linear(in_features=self._encoding_dim, out_features=self._encoding_dim,
                                              bias=False)
        self._model_layer_e = torch.nn.Linear(in_features=self._encoding_dim, out_features=self._encoding_dim,
                                              bias=False)
        # self._model_layer_s = torch.nn.Bilinear(in1_features=self._encoding_dim, in2_features=self._encoding_dim, out_features=)
        self._span_start_accuracy = CategoricalAccuracy()
        self._span_end_accuracy = CategoricalAccuracy()
        self._span_accuracy = BooleanAccuracy()
        self._squad_metrics = SquadEmAndF1()

        if dropout > 0:
            self._dropout = torch.nn.Dropout(p=dropout)
        else:
            self._dropout = lambda x: x

        self._mask_lstms = mask_lstms
        self._loss = torch.nn.CrossEntropyLoss()
        initializer(self)

    def forward(self, question: Dict[str, torch.LongTensor], passage: Dict[str, torch.LongTensor],
                span_start: torch.IntTensor = None, span_end: torch.IntTensor = None,
                metadata: List[Dict[str, Any]] = None) -> Dict[str, torch.Tensor]:

        # embedded_question = self._highway_layer(self._text_field_embedder(question))
        embedded_question = self._text_field_embedder(question)
        question_mask = util.get_text_field_mask(question).float()
        # embedded_passage = self._highway_layer(self._text_field_embedder(passage))
        embedded_passage = self._text_field_embedder(passage)
        passage_mask = util.get_text_field_mask(passage).float()

        batch_size = embedded_passage.size(0)
        passage_length = embedded_passage.size(1)
        question_length = embedded_question.size(1)

        question_lstm_mask = question_mask if self._mask_lstms else None
        passage_lstm_mask = passage_mask if self._mask_lstms else None

        # Shape(batch_size, question_length, encoding_dim)
        u_q = self._dropout(self._phrase_layer(embedded_question, question_lstm_mask))
        encoding_dim = u_q.size(-1)
        # Shape(batch_size, passage_length, encoding_dim)
        u_p = self._dropout(self._phrase_layer(embedded_passage, passage_lstm_mask))
        u_q = self._relu(self._atten_linear_layer(u_q))
        u_p = self._relu(self._atten_linear_layer(u_p))
        # Shape(batch_size, question_length, passage_length)
        # S_{ij} computes the similarity(attention weights)
        # between the i_th word of the question and the j_th word of the passage
        s = torch.bmm(u_q, u_p.transpose(2, 1))
        # Shape(batch_size, passage_length, encoding_dim)
        # P to Q
        q_ = attention_weight_sum_batch(util.masked_softmax(s.transpose(2, 1), passage_lstm_mask.unsqueeze(-1), dim=2),
                                        u_q)
        # Shape(batch_size, question_length, encoding_dim)
        # Q tot P
        p_ = attention_weight_sum_batch(util.masked_softmax(s, question_lstm_mask.unsqueeze(-1), dim=2), u_p)
        pp = self._atten_fusion(u_p, q_)
        # Shape(batch_size, question_length, encoding_dim)
        qq = self._atten_fusion(u_q, p_)
        # Shape(batch_size, passage_length, encoding_dim)
        d = self._passage_self_attention(pp, passage_lstm_mask)
        # Shape(batch_size, passage_length, encoding_dim)
        l = self._self_atten_layer(d, d)
        l = self._softmax_d2(l)
        # Shape(batch_size, passage_length, encoding_dim)
        d_ = l * d
        # Shape(batch_size, passage_length, encoding_dim)
        dd = self._self_atten_fusion(d, d_)
        # Shape(batch_size, passage_length, encoding_dim)
        ddd = self._semantic_rep_layer(dd, passage_lstm_mask)
        # Shape(batch_size, question_length, encoding_dim)
        qqq = self._contextual_question_layer(qq, question_lstm_mask)
        # Shape(batch_size, question_length, 1) -> (batch_size, question_length)
        # gamma = util.masked_softmax(self._vector_linear(qqq), question_lstm_mask.unsqueeze(-1), dim=2).squeeze(-1)
        qqq_tmp = self._vector_linear(qqq).squeeze(-1)
        gamma = self._softmax_d1(qqq_tmp)
        # Shape(batch_size, question_length)
        # (1, question_length) ` (question_length, encoding_dim)
        vec_q = torch.bmm(gamma.unsqueeze(1), qqq)
        # model & output layer
        # Shape(batch_size, 1, passage_length)
        vec_q_tmp = self._model_layer_s(vec_q)
        p_start = util.masked_softmax(torch.bmm(vec_q_tmp, ddd.transpose(2, 1)).squeeze(1),
                                      passage_lstm_mask, dim=1)
        # p_start = torch.bmm(vec_q_tmp, ddd.transpose(2, 1)).squeeze(1)
        # p_start = self._softmax_d1(p_start)
        span_start_logits = p_start
        # span_start_probs = util.masked_softmax(span_start_logits, passage_lstm_mask)
        # p_end = self._end_vector_matrix_bilinear(vec_q, ddd.permute(0, 2, 1))
        p_end = util.masked_softmax(torch.bmm(self._model_layer_e(vec_q), ddd.transpose(2, 1)).squeeze(1),
                                    passage_lstm_mask, dim=1)
        span_end_logits = p_end
        # span_start_logits = util.replace_masked_values(span_start_logits, passage_mask, 1e-7)
        # span_end_logits = util.replace_masked_values(span_end_logits, passage_mask, 1e-7)
        # span_end_probs = util.masked_softmax(span_end_logits, passage_lstm_mask)

        best_span = self.get_best_span(span_start_logits, span_end_logits)

        print("span_start_logits")
        print(span_start_logits)
        print("span_end_logits")
        print(span_end_logits)

        output = dict()
        output['best_span'] = best_span

        # Compute the loss for training
        if span_start is not None:
            # loss = nll_loss(util.masked_log_softmax(span_start_logits, passage_mask), span_start.squeeze(-1))
            # self._span_start_accuracy(span_start_logits, span_start.squeeze(-1))
            # loss += nll_loss(util.masked_log_softmax(span_end_logits, passage_mask), span_end.squeeze(-1))
            # self._span_end_accuracy(span_end_logits, span_end.squeeze(-1))
            # self._span_accuracy(best_span, torch.stack([span_start, span_end], -1))
            loss = self._loss(span_start_logits, span_start.squeeze(-1))
            self._span_start_accuracy(span_start_logits, span_start.squeeze(-1))
            loss += self._loss(span_end_logits, span_end.squeeze(-1))
            self._span_end_accuracy(span_end_logits, span_end.squeeze(-1))
            self._span_accuracy(best_span, torch.stack([span_start, span_end], -1))
            print(loss)
            output['loss'] = loss

        # Compute the EM and F1 on SQuAD and add the tokenized input to the output.
        if metadata is not None:
            output['best_span_str'] = []
            question_tokens = []
            passage_tokens = []
            for i in range(batch_size):
                question_tokens.append(metadata[i]['question_tokens'])
                passage_tokens.append(metadata[i]['passage_tokens'])
                passage_str = metadata[i]['original_passage']
                offsets = metadata[i]['token_offsets']
                predicted_span = tuple(best_span[i].detach().cpu().numpy())
                start_offset = offsets[predicted_span[0]][0]
                end_offset = offsets[predicted_span[1]][1]
                best_span_string = passage_str[start_offset:end_offset]
                output['best_span_str'].append(best_span_string)
                answer_texts = metadata[i].get('answer_texts', [])
                if answer_texts:
                    self._squad_metrics(best_span_string, answer_texts)
            output['question_tokens'] = question_tokens
            output['passage_tokens'] = passage_tokens
        return output

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        exact_match, f1_score = self._squad_metrics.get_metric(reset)
        return {
            'start_acc': self._span_start_accuracy.get_metric(reset),
            'end_acc': self._span_end_accuracy.get_metric(reset),
            'span_acc': self._span_accuracy.get_metric(reset),
            'em': exact_match,
            'f1': f1_score,
        }

    @staticmethod
    def get_best_span(span_start_logits: torch.Tensor, span_end_logits: torch.Tensor) -> torch.Tensor:
        if span_start_logits.dim() != 2 or span_end_logits.dim() != 2:
            raise ValueError("Input shapes must be (batch_size, passage_length)")
        batch_size, passage_length = span_start_logits.size()
        max_span_log_prob = [-1e20] * batch_size
        span_start_argmax = [0] * batch_size
        best_word_span = span_start_logits.new_zeros((batch_size, 2), dtype=torch.long)

        span_start_logits = span_start_logits.detach().cpu().numpy()
        span_end_logits = span_end_logits.detach().cpu().numpy()

        for b in range(batch_size):  # pylint: disable=invalid-name
            for j in range(passage_length):
                val1 = span_start_logits[b, span_start_argmax[b]]
                if val1 < span_start_logits[b, j]:
                    span_start_argmax[b] = j
                    val1 = span_start_logits[b, j]

                val2 = span_end_logits[b, j]

                if val1 + val2 > max_span_log_prob[b]:
                    best_word_span[b, 0] = span_start_argmax[b]
                    best_word_span[b, 1] = j
                    max_span_log_prob[b] = val1 + val2
        return best_word_span
