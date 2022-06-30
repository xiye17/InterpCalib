from transformers import RobertaTokenizer
import string

def _merge_roberta_tokens_into_words(tokenizer, feature):
    tokens = tokenizer.convert_ids_to_tokens(feature.input_ids)
    decoded_each_tok = [
        bytearray([tokenizer.byte_decoder[c] for c in t]).decode("utf-8", errors=tokenizer.errors) for t in tokens
    ]


    end_points = []
    force_break = False
    for i, t in enumerate(decoded_each_tok):
        # special token
        if t in tokenizer.all_special_tokens:
            end_points.append(i)
            force_break = True
            continue

         # no alphanum
        if not any([x.isalnum() for x in t.lstrip()]):
            end_points.append(i)
            force_break = True
            continue

        if not t.lstrip()[0].isalnum():
            # print('Special force', t)
            end_points.append(i)
            force_break = True
            continue

        if t in string.punctuation:
            end_points.append(i)
            force_break = True
            continue

        if force_break:
            end_points.append(i)
            force_break = False
            continue

        # if in question segment
        if t[0] == ' ':
            decoded_each_tok[i] = t[1:]
            end_points.append(i)
    end_points.append(len(decoded_each_tok))

    # if in context segment
    segments = []
    for i in range(1, len(end_points)):
        if end_points[i - 1] == end_points[i]:
            continue
        segments.append((end_points[i - 1], end_points[i]))
    
    merged_tokens = []
    for s0, s1 in segments:
        merged_tokens.append(''.join(decoded_each_tok[s0:s1]))
    return merged_tokens, segments

def merge_tokens_into_words(tokenizer, feature):
    if isinstance(tokenizer, RobertaTokenizer):
        return _merge_roberta_tokens_into_words(tokenizer, feature)