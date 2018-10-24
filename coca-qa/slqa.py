import logging
import torch
from torch.nn.parameter import Parameter
from torch.nn.functional import nll_loss
from typing import Optional, Dict, List, Any

from allennlp.models.model import Model
from allennlp.data import Vocabulary
from allennlp.modules import TextFieldEmbedder, Seq2SeqEncoder, FeedForward
from allennlp.modules.matrix_attention.bilinear_matrix_attention import BilinearMatrixAttention
from allennlp.nn import RegularizerApplicator, InitializerApplicator
from allennlp.training.metrics import CategoricalAccuracy, BooleanAccuracy
from models.vector_matrix_bilinear import VectorMatrixLinear
from models.vector_linear import VectorLinear
from utils.vector_weight_sum import vector_weight_sum_matrix, vector_weight_sum

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


@Model.register("slqa")
class MultiGranularityHierarchicalAttentionFusionNetworks(Model):

    def __init__(self, vocab: Vocabulary,
                 text_field_embedder: TextFieldEmbedder,
                 passage_bilstm_encoder: Seq2SeqEncoder,
                 question_bilstm_encoder: Seq2SeqEncoder,
                 passage_self_attention: Seq2SeqEncoder,
                 question_self_attention: Seq2SeqEncoder,
                 passage_matrix_attention: BilinearMatrixAttention,
                 semantic_rep_layer: Seq2SeqEncoder,
                 contextual_question_layer: Seq2SeqEncoder,
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
        self._fuse_linear_d = torch.nn.Linear(in_features=4 * self._passage_self_attention.get_output_dim(),
                                              out_features=self._passage_self_attention.get_output_dim())

        self._passage_self_attention = passage_self_attention
        self._question_self_attention = question_self_attention
        self._passage_matrix_attention = passage_matrix_attention
        self._passage_matrix_attention_softmax = torch.nn.Softmax(dim=1)

        self._w1 = Parameter(torch.Tensor(self._passage_self_attention.get_output_dim(), ))

        self._semantic_rep_layer = semantic_rep_layer
        self._contextual_question_layer = contextual_question_layer

        self._vector_linear = VectorLinear(self._contextual_question_layer.get_output_dim())
        self._vector_matrix_bilinear = VectorMatrixLinear(self._contextual_question_layer.get_output_dim())

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
        batch_size = embedded_passage.size(0)
        # Shape(batch_size, question_length, encoding_dim)
        u_q = self._question_bilstm_encoder(embedded_question)
        # Shape(batch_size, passage_length, encoding_dim)
        u_p = self._passage_bilstm_encoder(embedded_passage)
        # Shape(batch_size, question_length, passage_length)
        s = torch.mm(self._linear_activate(self._atten_linear_layer(u_q)), self._linear_activate(
            self._atten_linear_layer(u_p)).permute(0, 2, 1))
        # Shape(batch_size, encoding_dim, passage_length)
        q_ = vector_weight_sum_matrix(self._softmax_d1(s).permute(0, 2, 1), u_q.permute(0, 2, 1))
        # Shape(batch_size, encoding_dim, question_length)
        p_ = vector_weight_sum_matrix(self._softmax_d2(s), u_p.permute(0, 2, 1))
        # Shape(batch_size, question_length, encoding_dim)
        p_ = p_.permute(0, 2, 1)
        # Shape(batch_size, passage_length, encoding_dim)
        q_ = q_.permute(0, 2, 1)
        # Shape(batch_size, passage_length, 4 * encoding_dim)
        p_q_ = torch.cat((u_p, q_, u_p * q_, u_p - q_), 2)
        # Shape(batch_size, question_length, 4 * encoding_dim)
        q_p_ = torch.cat((u_q, p_, u_q * p_, u_q - p_), 2)
        # Shape(batch_size, passage_length, passage_length)
        pp = torch.mm(self._fuse_sigmoid(self._fuse_linear_g(p_q_)),
                      self._fuse_tanh(self._fuse_linear_m(p_q_)).permute(0, 2, 1)) + torch.mm(
            (torch.Tensor([1]) - self._fuse_sigmoid(self._fuse_linear_g(p_q_))), u_p.permute(0, 2, 1))
        # Shape(batch_size, question_length, question_length)
        qq = torch.mm(self._fuse_sigmoid(self._fuse_linear_g(q_p_)),
                      self._fuse_tanh(self._fuse_linear_m(q_p_)).permute(0, 2, 1)) + torch.mm(
            (torch.Tensor([1]) - self._fuse_sigmoid(self._fuse_linear_g(q_p_))), u_q.permute(0, 2, 1))
        # Shape(batch_size, passage_length, encoding_dim_1)
        d = self._passage_self_attention(pp)
        # Shape(batch_size, passage_length, passage_length)
        l = self._passage_matrix_attention(d, d)
        tmp = l.size(1)
        l = self._passage_matrix_attention_softmax(l.view(batch_size, -1)).view(batch_size, tmp, -1)
        # Shape(batch_size, passage_length, encoding_dim_1)
        d_ = torch.mm(l, d)
        # simple fuse function
        d_d_ = torch.cat((d, d_, d * d_, d - d_), 2)
        dd = self._fuse_tanh(self._fuse_linear_d(d_d_))
        # Shape(batch_size, passage_length, encoding_dim_2)
        ddd = self._semantic_rep_layer(dd)
        # Shape(batch_size, question_length, question_length)
        qqq = self._contextual_question_layer(qq)
        # Shape(batch_size, question_length, 1)
        gamma = self._vector_linear(qqq)
        # Shape(batch_size, question_length)
        vec_q = vector_weight_sum(gamma, qqq)

        # model & output layer
        # Shape(batch_size, 1, encoding_dim_2)
        p_start = self._vector_matrix_bilinear(vec_q, ddd)
        p_end = self._vector_matrix_bilinear(vec_q, ddd)
