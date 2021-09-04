def prepare_train_features(examples, tokenizer):

    contexts = [examples[i]['context'] for i in range(len(examples))]
    questions = [examples[i]['question'] for i in range(len(examples))]

    tokenized_examples = tokenizer(questions, contexts,
                                   stride=128, max_seq_len=128)

    for i, tokenized_example in enumerate(tokenized_examples):
        # We will label impossible answers with the index of the CLS token.
        input_ids = tokenized_example["input_ids"]
        cls_index = input_ids.index(tokenizer.cls_token_id)

        # The offset mappings will give us a map from token to character position in the original context. This will
        # help us compute the start_positions and end_positions.
        offsets = tokenized_example['offset_mapping']

        # Grab the sequence corresponding to that example (to know what is the context and what is the question).
        sequence_ids = tokenized_example['token_type_ids']

        # One example can give several spans, this is the index of the example containing this span of text.
        sample_index = tokenized_example['overflow_to_sample']
        answers = examples[sample_index]['answers']
        answer_starts = examples[sample_index]['answer_starts']

        # If no answers are given, set the cls_index as answer.
        if len(answer_starts) == 0:
            tokenized_examples[i]["start_positions"] = cls_index
            tokenized_examples[i]["end_positions"] = cls_index
        else:
            # Start/end character index of the answer in the text.
            start_char = answer_starts[0]
            end_char = start_char + len(answers[0])

            # Start token index of the current span in the text.
            token_start_index = 0
            while sequence_ids[token_start_index] != 1:
                token_start_index += 1

            # End token index of the current span in the text.
            token_end_index = len(input_ids) - 1
            while sequence_ids[token_end_index] != 1:
                token_end_index -= 1
            # Minus one more to reach actual text
            token_end_index -= 1

            # Detect if the answer is out of the span (in which case this feature is labeled with the CLS index).
            if not (offsets[token_start_index][0] <= start_char and
                    offsets[token_end_index][1] >= end_char):
                tokenized_examples[i]["start_positions"] = cls_index
                tokenized_examples[i]["end_positions"] = cls_index
            else:
                # Otherwise move the token_start_index and token_end_index to the two ends of the answer.
                # Note: we could go after the last offset if the answer is the last word (edge case).
                while token_start_index < len(offsets) and offsets[
                        token_start_index][0] <= start_char:
                    token_start_index += 1
                tokenized_examples[i]["start_positions"] = token_start_index - 1
                while offsets[token_end_index][1] >= end_char:
                    token_end_index -= 1
                tokenized_examples[i]["end_positions"] = token_end_index + 1

    return tokenized_examples

