import json
import logging
from typing import Dict, List, Tuple, Any

from overrides import overrides

from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.instance import Instance
from allennlp.data.dataset_readers.reading_comprehension import util
from allennlp.data.token_indexers import SingleIdTokenIndexer, TokenIndexer
from allennlp.data.tokenizers import Token, Tokenizer, WordTokenizer
from allennlp.data.fields import Field, TextField, IndexField, \
    MetadataField, LabelField, ListField

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


@DatasetReader.register("cocaqa")
class CocaQAReader(DatasetReader):

    def __init__(self,
                 tokenizer: Tokenizer = None,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 lazy: bool = False) -> None:
        super().__init__(lazy)
        self._tokenizer = tokenizer or WordTokenizer()
        self._token_indexers = token_indexers or {'tokens': SingleIdTokenIndexer()}

    @overrides
    def _read(self, file_path: str):
        # if `file_path` is a URL, redirect to the cache
        file_path = cached_path(file_path)
        logger.info("Reading file at %s", file_path)
        with open(file_path) as dataset_file:
            dataset_json = json.load(dataset_file)
            dataset = dataset_json['data']
        logger.info("Reading the dataset")

        debug = 0

        for paragraph_json in dataset:
            paragraph = paragraph_json["story"]
            tokenized_paragraph = self._tokenizer.tokenize(paragraph)
            qas = paragraph_json['questions']
            metadata = {}
            metadata["id"] = paragraph_json["id"]
            metadata["instance_id"] = [qa['turn_id'] for qa in qas]
            # question_text_list = [qa["input_text"].strip().replace("\n", "") for qa in qas]
            question_text_list = []
            # answer_texts_list = [[answer['text'] for answer in qa['answers']] for qa in qas]
            questions = paragraph_json['questions']
            answers = paragraph_json['answers']
            for i, (question, answer) in enumerate(zip(questions, answers)):
                q_text = question['input_text']
                if i > 0:
                    q_text = questions[i - 1]['input_text'] + answers[i - 1]['input_text'] + q_text
                if i > 1:
                    q_text = questions[i - 2]['input_text'] + answers[i - 2]['input_text'] + q_text
                if i > 2:
                    q_text = questions[i - 3]['input_text'] + answers[i - 3]['input_text'] + q_text
                question_text_list.append(q_text)
            answer_texts_list = list()
            span_starts_list = list()
            span_ends_list = list()
            yesno_list = list()
            for answer in answers:
                answer_text_list = list()
                span_start_list = list()
                span_end_list = list()
                span_text = answer['span_text']
                input_text = answer['input_text'].strip().replace("\n", "")
                before = self.get_front_blanks(span_text, 0)
                span_text = span_text.strip().replace("\n", "")
                beg = span_text.find(input_text)
                span_start = answer['span_start'] + before
                span_end = span_start + len(span_text)

                if input_text.lower() == "unknown":
                    span_start = 0
                    span_end = 0
                    input_text = paragraph[0]
                    yesno_list.append("x")
                    answer_text = input_text
                elif input_text.lower() == "yes":
                    yesno_list.append("y")
                    answer_text = span_text
                elif input_text.lower() == "no":
                    yesno_list.append("n")
                    answer_text = span_text
                else:
                    yesno_list.append("x")
                    answer_text = input_text
                    if beg != -1:
                        span_start = span_start + beg
                        span_end = span_start + len(input_text)
                        debug = debug + 1

                # debug 11.9 0.23
                # answer_text = input_text

                answer_text_list.append(answer_text)
                span_start_list.append(span_start)
                span_end_list.append(span_end)

                span_starts_list.append(span_start_list)
                span_ends_list.append(span_end_list)
                answer_texts_list.append(answer_text_list)

            if "additional_answers" in paragraph_json:
                for key in paragraph_json['additional_answers']:
                    for additional_answer in paragraph_json['additional_answers'][key]:
                        input_text = additional_answer['input_text'].strip().replace("\n", "")
                        span_text = additional_answer['span_text']
                        before = self.get_front_blanks(span_text, 0)
                        span_text = span_text.strip().replace("\n", "")
                        beg = span_text.find(input_text)
                        span_start = additional_answer['span_start'] + before
                        span_end = span_start + len(span_text)

                        if input_text.lower() == "unknown":
                            span_start = 0
                            span_end = 0
                            input_text = paragraph[0]
                            answer_text = input_text
                        elif input_text.lower() == "yes":
                            answer_text = span_text
                        elif input_text.lower() == "no":
                            answer_text = span_text
                        else:
                            answer_text = input_text
                            if beg != -1:
                                span_start = span_start + beg
                                span_end = span_start + len(input_text)
                                debug = debug + 1

                        question_id = additional_answer['turn_id'] - 1
                        span_starts_list[question_id].append(span_start)
                        span_ends_list[question_id].append(span_end)
                        answer_texts_list[question_id].append(answer_text)

            metadata["question"] = question_text_list
            metadata['answer_texts_list'] = answer_texts_list
            instance = self.text_to_instance(question_text_list,
                                             paragraph,
                                             span_starts_list,
                                             span_ends_list,
                                             tokenized_paragraph,
                                             yesno_list,
                                             metadata)
            yield instance

        print("debug")
        print(debug)

    def get_front_blanks(self, answer, padding):
        answer = answer.replace("\n", "")
        before = 0
        for i in range(len(answer)):
            if answer[i] == ' ':
                before += 1
            else:
                break
        return before - padding

    @overrides
    def text_to_instance(self,  # type: ignore
                         question_text_list: List[str],
                         passage_text: str,
                         start_span_list: List[List[int]] = None,
                         end_span_list: List[List[int]] = None,
                         passage_tokens: List[Token] = None,
                         yesno_list: List[str] = None,
                         additional_metadata: Dict[str, Any] = None) -> Instance:
        # pylint: disable=arguments-differ
        # We need to convert character indices in `passage_text` to token indices in
        # `passage_tokens`, as the latter is what we'll actually use for supervision.
        answer_token_span_list = []
        passage_offsets = [(token.idx, token.idx + len(token.text)) for token in passage_tokens]
        for start_list, end_list in zip(start_span_list, end_span_list):
            token_spans: List[Tuple[int, int]] = []
            for char_span_start, char_span_end in zip(start_list, end_list):
                (span_start, span_end), error = util.char_span_to_token_span(passage_offsets,
                                                                             (char_span_start, char_span_end))
                if error:
                    logger.debug("Passage: %s", passage_text)
                    logger.debug("Passage tokens: %s", passage_tokens)
                    logger.debug("Answer span: (%d, %d)", char_span_start, char_span_end)
                    logger.debug("Token span: (%d, %d)", span_start, span_end)
                    logger.debug("Tokens in answer: %s", passage_tokens[span_start:span_end + 1])
                    logger.debug("Answer: %s", passage_text[char_span_start:char_span_end])
                token_spans.append((span_start, span_end))
            answer_token_span_list.append(token_spans)
        question_list_tokens = [self._tokenizer.tokenize(q) for q in question_text_list]
        # Map answer texts to "CANNOTANSWER" if more than half of them marked as so.
        additional_metadata['answer_texts_list'] = [util.handle_cannot(ans_list) for ans_list \
                                                    in additional_metadata['answer_texts_list']]
        return self.make_reading_comprehension_instance_quac(question_list_tokens,
                                                             passage_tokens,
                                                             self._token_indexers,
                                                             passage_text,
                                                             answer_token_span_list,
                                                             yesno_list,
                                                             additional_metadata)

    def make_reading_comprehension_instance_quac(self,
                                                 question_list_tokens: List[List[Token]],
                                                 passage_tokens: List[Token],
                                                 token_indexers: Dict[str, TokenIndexer],
                                                 passage_text: str,
                                                 token_span_lists: List[List[Tuple[int, int]]] = None,
                                                 yesno_list: List[int] = None,
                                                 additional_metadata: Dict[str, Any] = None) -> Instance:
        additional_metadata = additional_metadata or {}
        fields: Dict[str, Field] = {}
        passage_offsets = [(token.idx, token.idx + len(token.text)) for token in passage_tokens]
        # This is separate so we can reference it later with a known type.
        passage_field = TextField(passage_tokens, token_indexers)
        fields['passage'] = passage_field
        fields['question'] = ListField([TextField(q_tokens, token_indexers) for q_tokens in question_list_tokens])
        metadata = {'original_passage': passage_text,
                    'token_offsets': passage_offsets,
                    'question_tokens': [[token.text for token in question_tokens] \
                                        for question_tokens in question_list_tokens],
                    'passage_tokens': [token.text for token in passage_tokens], }
        if token_span_lists:
            span_start_list: List[Field] = []
            span_end_list: List[Field] = []
            for question_index, answer_span_lists in enumerate(token_span_lists):
                span_start, span_end = min(answer_span_lists, key=lambda x: x[1] - x[0])
                span_start_list.append(IndexField(span_start, passage_field))
                span_end_list.append(IndexField(span_end, passage_field))

            fields['span_start'] = ListField(span_start_list)
            fields['span_end'] = ListField(span_end_list)
            fields['yesno_list'] = ListField(
                [LabelField(yesno, label_namespace="yesno_labels") for yesno in yesno_list])
        metadata.update(additional_metadata)
        fields['metadata'] = MetadataField(metadata)
        return Instance(fields)