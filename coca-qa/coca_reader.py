import json
import logging
from typing import Dict, List, Tuple

from overrides import overrides

from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.instance import Instance
from allennlp.data.dataset_readers.reading_comprehension import util
from allennlp.data.token_indexers import SingleIdTokenIndexer, TokenIndexer
from allennlp.data.tokenizers import Token, Tokenizer, WordTokenizer

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


@DatasetReader.register("cocaqa")
class CocaQAReader(DatasetReader):
    """
    Reads a JSON-formatted SQuAD file and returns a ``Dataset`` where the ``Instances`` have four
    fields: ``question``, a ``TextField``, ``passage``, another ``TextField``, and ``span_start``
    and ``span_end``, both ``IndexFields`` into the ``passage`` ``TextField``.  We also add a
    ``MetadataField`` that stores the instance's ID, the original passage text, gold answer strings,
    and token offsets into the original passage, accessible as ``metadata['id']``,
    ``metadata['original_passage']``, ``metadata['answer_texts']`` and
    ``metadata['token_offsets']``.  This is so that we can more easily use the official SQuAD
    evaluation script to get metrics.
    Parameters
    ----------
    tokenizer : ``Tokenizer``, optional (default=``WordTokenizer()``)
        We use this ``Tokenizer`` for both the question and the passage.  See :class:`Tokenizer`.
        Default is ```WordTokenizer()``.
    token_indexers : ``Dict[str, TokenIndexer]``, optional
        We similarly use this for both the question and the passage.  See :class:`TokenIndexer`.
        Default is ``{"tokens": SingleIdTokenIndexer()}``.
    """

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
        for paragraph_json in dataset:
            paragraph = paragraph_json["story"]
            # paragraph = paragraph_json["story"].strip().replace("\n", "")
            n_paragraph, padding = self.delete_leading_tokens_of_paragraph(paragraph)
            # tokenized_paragraph = self._tokenizer.tokenize(paragraph)
            tokenized_paragraph = self._tokenizer.tokenize(n_paragraph)

            # store the history
            # append previous answers to the passage and the previous questions
            history = list()

            ind = 0
            for question_answer in paragraph_json['questions']:
                question_text = question_answer["input_text"].strip().replace("\n", "")
                # question_text = question_answer["input_text"].replace("\n", "")
                answer_texts = []

                tmp = paragraph_json["answers"][ind]['span_text']
                before = self.get_front_blanks(tmp, padding)
                answer = paragraph_json["answers"][ind]['span_text'].strip().replace("\n", "")
                start = paragraph_json["answers"][ind]['span_start'] + before
                end = start + len(answer)

                # debug 10.15 21:20
                if answer.lower() == "unknown":
                    answer = n_paragraph[0]
                    start = 0
                    end = 0

                answer_texts.append(answer)
                # answer_texts = [answer['text'] for answer in question_answer['answers']]

                history.append((question_text, answer))

                span_starts = list()
                span_starts.append(start)

                span_ends = list()
                span_ends.append(end)
                # span_starts = [answer['answer_start'] for answer in question_answer['answers']]
                # span_ends = [start + len(answer) for start, answer in zip(span_starts, answer_texts)]

                if "additional_answers" in paragraph_json:
                    additional_answers = paragraph_json["additional_answers"]
                    for key in additional_answers:
                        tmp = additional_answers[key][ind]["span_text"]
                        answer = tmp.strip().replace("\n", "")
                        before = self.get_front_blanks(tmp, padding)
                        start = additional_answers[key][ind]["span_start"] + before
                        end = start + len(answer)

                        # debug 10.15 21:20
                        if answer.lower() == "unknown":
                            answer = n_paragraph[0]
                            start = 0
                            end = 0

                        answer_texts.append(answer)
                        span_starts.append(start)
                        span_ends.append(end)

                his_paragraph = paragraph
                his_question = question_text
                if ind > 1:
                    his_paragraph = his_paragraph + " " + str(history[ind - 1][1])
                    his_question = str(history[ind - 1][0]) + " " + his_question
                    if ind > 2:
                        his_paragraph = his_paragraph + " " + str(history[ind - 2][1])
                        his_question = str(history[ind - 2][0]) + " " + his_question
                his_tokenized_paragraph = self._tokenizer.tokenize(his_paragraph)

                ind += 1

                instance = self.text_to_instance(his_question,
                                                 his_paragraph,
                                                 zip(span_starts, span_ends),
                                                 answer_texts,
                                                 his_tokenized_paragraph)
                yield instance

    def get_front_blanks(self, answer, padding):
        answer = answer.replace("\n", "")
        before = 0
        for i in range(len(answer)):
            if answer[i] == ' ':
                before += 1
            else:
                break
        return before - padding

    def delete_leading_tokens_of_paragraph(self, paragraph):
        before = 0
        for i in range(len(paragraph)):
            if paragraph[i] == ' ' or paragraph[i] == '\n':
                before += 1
            else:
                break

        nparagraph = paragraph[before:]
        return nparagraph, before

    @overrides
    def text_to_instance(self,  # type: ignore
                         question_text: str,
                         passage_text: str,
                         char_spans: List[Tuple[int, int]] = None,
                         answer_texts: List[str] = None,
                         passage_tokens: List[Token] = None) -> Instance:
        # pylint: disable=arguments-differ
        if not passage_tokens:
            passage_tokens = self._tokenizer.tokenize(passage_text)
        char_spans = char_spans or []

        # We need to convert character indices in `passage_text` to token indices in
        # `passage_tokens`, as the latter is what we'll actually use for supervision.
        token_spans: List[Tuple[int, int]] = []
        passage_offsets = [(token.idx, token.idx + len(token.text)) for token in passage_tokens]
        for char_span_start, char_span_end in char_spans:
            (span_start, span_end), error = util.char_span_to_token_span(passage_offsets,
                                                                         (char_span_start, char_span_end))
            if error:
                logger.debug("Passage: %s", passage_text)
                logger.debug("Passage tokens: %s", passage_tokens)
                logger.debug("Question text: %s", question_text)
                logger.debug("Answer span: (%d, %d)", char_span_start, char_span_end)
                logger.debug("Token span: (%d, %d)", span_start, span_end)
                logger.debug("Tokens in answer: %s", passage_tokens[span_start:span_end + 1])
                logger.debug("Answer: %s", passage_text[char_span_start:char_span_end])
            token_spans.append((span_start, span_end))

        return util.make_reading_comprehension_instance(self._tokenizer.tokenize(question_text),
                                                        passage_tokens,
                                                        self._token_indexers,
                                                        passage_text,
                                                        token_spans,
                                                        answer_texts)

