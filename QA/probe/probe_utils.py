import numpy as np
import torch

def stats_of_layer_attribution(attribution, feature):
    tokens = feature.tokens

    # pool
    attribution = attribution.numpy()
    attribution = np.sum(attribution, axis=0)
    
    num_tokens = attribution.shape[0]
    attribution_abs = np.abs(attribution)    
    attribution_abs = attribution_abs.flatten()
    attribution_abs = attribution_abs / np.sum(attribution_abs)
    sorted_idx = np.argsort(-attribution_abs)    
    attribution_abs_sorted = attribution_abs[sorted_idx]

    top_thresholds = [0.005, 0.01, 0.02]
    
    for threshold in top_thresholds:
        topk = int(threshold * attribution_abs_sorted.size)
        percent_pertained = np.sum(attribution_abs_sorted[:topk])
        print('% Threshould: {:.1f}, Retained {:.1f}'.format(threshold * 100, percent_pertained * 100))
        
        selected_index = sorted_idx[:topk]
        attn_src = np.floor_divide(selected_index, num_tokens)
        attn_dst = np.remainder(selected_index, num_tokens)
        attn_src = np.sort(np.unique(attn_src))
        attn_dst = np.sort(np.unique(attn_dst))
        
        print('src involved', attn_src.size, [tokens[i] for i in attn_src])
        print('dst involved', attn_dst.size, [tokens[i] for i in attn_dst])

        # print('src involved', attn_src.size, 'dst involved', attn_dst.size)



def get_link_mask_by_thresholds(attribution, top_thresholds, pad_to_len=0):
    num_heads = attribution.shape[0]
    num_tokens = attribution.shape[1]

    attribution = attribution.numpy()
    attribution = np.sum(attribution, axis=0)
    attribution[attribution<0] = 0
    attribution_abs = np.abs(attribution)    
    attribution_abs = attribution_abs.flatten()
    sorted_idx = np.argsort(-attribution_abs)    
    
    return_link_masks = []
    np_bool_type = np.dtype(bool)
    for threshold in top_thresholds:
        topk = int(threshold * attribution_abs.size)
        selected_index = sorted_idx[:topk]

        link_mask = np.zeros_like(sorted_idx, dtype=np_bool_type)
        link_mask[selected_index] = True
        link_mask = np.reshape(link_mask, (num_tokens, num_tokens))

        # link_mask[np.arange(num_tokens),np.arange(num_tokens)] = True
        # print(link_mask.shape, topk, np.sum(link_mask))                
        if pad_to_len > 0:
            padded_mask = np.ones((pad_to_len, pad_to_len), dtype=np_bool_type)
            padded_mask[:num_tokens, :num_tokens] = link_mask
        else:
            padded_mask = link_mask

        padded_mask = torch.BoolTensor(padded_mask)
        padded_mask = padded_mask.expand(1, num_heads, -1, -1)
        return_link_masks.append(padded_mask)

    return return_link_masks



def get_link_mask_by_token_thresholds(attribution, top_thresholds, pad_to_len=0):
    num_heads = attribution.shape[0]
    num_tokens = attribution.shape[1]

    attribution = attribution.numpy()
    attribution = np.sum(attribution, axis=0)
    attribution[attribution<0] = 0

    token_values = np.sum(attribution, axis=0) + np.sum(attribution, axis=1)    
    sorted_idx = np.argsort(-token_values)    
    
    return_link_masks = []
    np_bool_type = np.dtype(bool)
    for threshold in top_thresholds:
        topk = int(threshold * token_values.size)
        selected_index = sorted_idx[:topk]
        
        mask_row = np.zeros_like(sorted_idx, dtype=np_bool_type)
        mask_row[selected_index] = True
        link_mask = np.zeros([num_tokens, num_tokens], dtype=np_bool_type)
        link_mask[selected_index] = mask_row        
        # link_mask[np.arange(num_tokens),np.arange(num_tokens)] = True
        # print(link_mask.shape, topk, np.sum(link_mask))                
        if pad_to_len > 0:
            padded_mask = np.ones((pad_to_len, pad_to_len), dtype=np_bool_type)
            padded_mask[:num_tokens, :num_tokens] = link_mask
        else:
            padded_mask = link_mask

        padded_mask = torch.BoolTensor(padded_mask)
        padded_mask = padded_mask.expand(1, num_heads, -1, -1)
        return_link_masks.append(padded_mask)

    return return_link_masks



        