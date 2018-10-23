import logging
import torch
from torch.nn.functional import nll_loss
from typing import Optional, Dict, List, Any

from allennlp.models.model import Model
from allennlp.data import Vocabulary
from allennlp.modules import TextFieldEmbedder, Seq2SeqEncoder, FeedForward
from allennlp.nn import RegularizerApplicator, InitializerApplicator
from allennlp.training.metrics import CategoricalAccuracy, BooleanAccuracy

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


@Model.register("slqa")
class MultiGranularityHierarchicalAttentionFusionNetworks(Model):

    def __init__(self, vocab: Vocabulary,
                 text_field_embedder: TextFieldEmbedder,
                 passage_bilstm_encoder: Seq2SeqEncoder,
                 question_bilstm_encoder: Seq2SeqEncoder,
                 dropout: float = 0.2,
                 regularizer: Optional[RegularizerApplicator] = None,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 ):

        super(MultiGranularityHierarchicalAttentionFusionNetworks, self).__init__(vocab, regularizer)
        self._text_field_embedder = text_field_embedder
        self._passage_bilstm_encoder = passage_bilstm_encoder
        self._question_bilstm_encoder = question_bilstm_encoder
        self._encoding_dim = self._passage_bilstm_encoder.get_output_dim()
        self._atten_linear_layer = torch.nn.Linear(in_features=self._encoding_dim,
                                                   out_features=self._encoding_dim, bias=False)
        self._linear_activate = torch.nn.ReLU()
        self._softmax_d1 = torch.nn.Softmax(dim=1)
        self._softmax_d2 = torch.nn.Softmax(dim=2)
        self._fuse_linear_m = torch.nn.Linear(in_features=4 * self._encoding_dim, out_features=self._encoding_dim)
        self._fuse_linear_g = torch.nn.Linear(in_features=4 * self._encoding_dim, out_features=self._encoding_dim)
        self._fuse_tanh = torch.nn.Tanh()
        self._fuse_sigmoid = torch.nn.Sigmoid()

        self._span_start_accuracy = CategoricalAccuracy()
        self._span_end_accuracy = CategoricalAccuracy()
        self._span_accuracy = BooleanAccuracy()
        self._squad_metrics = SquadEmAndF1()

        if dropout > 0:
            self._dropout = torch.nn.Dropout(p=dropout)
        else:
            self._dropout = lambda x: x

        initializer(self)

    def forward(self, question: Dict[str, torch.LongTensor], passage: Dict[str, torch.LongTensor],
                span_start: torch.IntTensor = None, span_end: torch.IntTensor = None,
                metadata: List[Dict[str, Any]] = None) -> Dict[str, torch.Tensor]:

        embedded_question = self._text_field_embedder(question)
        embedded_passage = self._text_field_embedder(passage)

        ## TODO:mask
        # Shape(batch_size, question_length, encoding_dim)
        u_q = self._question_bilstm_encoder(embedded_question)
        # Shape(batch_size, passage_length, encoding_dim)
        u_p = self._passage_bilstm_encoder(embedded_passage)
        # Shape(batch_size, question_length, passage_length)
        s = torch.mm(self._linear_activate(self._atten_linear_layer(u_q)), self._linear_activate(
            self._atten_linear_layer(u_p)).permute(0, 2, 1))
        # Shape(batch_size, passage_length, encoding_dim)
        q_ = torch.mm(self._softmax_d1(s), u_q)
        # Shape(batch_size, question_length, encoding_dim)
        p_ = torch.mm(self._softmax_d2(s), u_p)
        p_q_ = torch.cat((u_p, q_, u_p * q_, u_p - q_), 2)
        q_p_ = torch.cat((u_q, p_, u_q * p_, u_q - p_), 2)
        # Shape(batch_size, passage_length, passage_length)
        pp = torch.mm(self._fuse_sigmoid(self._fuse_linear_g(p_q_)),
                      self._fuse_tanh(self._fuse_linear_m(p_q_)).permute(0, 2, 1)) + torch.mm(
            (torch.Tensor([1]) - self._fuse_sigmoid(self._fuse_linear_g(p_q_))), u_p.permute(0, 2, 1))
        # Shape(batch_size, question_length, question_length)
        qq = torch.mm(self._fuse_sigmoid(self._fuse_linear_g(q_p_)),
                      self._fuse_tanh(self._fuse_linear_m(q_p_)).permute(0, 2, 1)) + torch.mm(
            (torch.Tensor([1]) - self._fuse_sigmoid(self._fuse_linear_g(q_p_))), u_q.permute(0, 2, 1))
