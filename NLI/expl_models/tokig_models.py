from transformers import RobertaForSequenceClassification
from transformers.modeling_roberta import (
    RobertaClassificationHead,
    RobertaPreTrainedModel,
    RobertaModel,
    RobertaEncoder,
    RobertaAttention,
    RobertaSelfAttention,
    RobertaEmbeddings,
    RobertaAttention,
    RobertaSelfOutput,
    RobertaIntermediate,
    RobertaOutput,
    RobertaLayer,
    create_position_ids_from_input_ids,
)

from transformers.modeling_utils import (
    PreTrainedModel,
    apply_chunking_to_forward,
    find_pruneable_heads_and_indices,
    prune_linear_layer,
)
import torch
from torch import nn
import torch.nn.functional as F
import math


class TokIGRobertaForSequenceClassification(RobertaPreTrainedModel):
    authorized_unexpected_keys = [r"pooler"]
    authorized_missing_keys = [r"position_ids"]

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.roberta = RobertaModel(config, add_pooling_layer=False)
        self.classifier = RobertaClassificationHead(config)

        self.init_weights()

    def forward(
        self,
        do_attribute=False,
        **kwargs,
    ):
        if do_attribute:
            return self.attribute(**kwargs)
        else:
            return self.predict(**kwargs)

    def predict(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the sequence classification/regression loss.
            Indices should be in :obj:`[0, ..., config.num_labels - 1]`.
            If :obj:`config.num_labels == 1` a regression loss is computed (Mean-Square loss),
            If :obj:`config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = outputs[0]
        logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
    
    def _get_origin_emb(self, input_ids):
        return self.roberta.embeddings.word_embeddings(input_ids)
    
    def _get_origin_position_ids(self, input_ids):
        return create_position_ids_from_input_ids(input_ids, self.config.pad_token_id).to(input_ids.device)

    @staticmethod
    def probs_of_pred(final_logits, pred_indexes):
        final_logits = F.softmax(final_logits, dim=1)
        selected_logits = torch.gather(final_logits, 1, pred_indexes.view(-1, 1)).squeeze(1)
        return selected_logits

    def attribute(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,       
        pred_indexes=None,
        final_logits=None, # for comparison        
        num_steps=300
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        acc_attribution = .0                 
        # input size B * L
        # attention mask B * L
        # N_Layer * B * N_HEAD * L * L
        # TODO: FIX hard encoding
        mask_token_id = 50264
        baseline_input_ids = mask_token_id * torch.ones_like(input_ids)        
        baseline_embs = self._get_origin_emb(baseline_input_ids)
        target_embs = self._get_origin_emb(input_ids)
        target_position_ids = self._get_origin_position_ids(input_ids)
        diff_embs = target_embs - baseline_embs        
        for step_i in range(num_steps):    
            # compose input
            step_embs = diff_embs * step_i / num_steps + baseline_embs
            step_embs.requires_grad_(True)
            outputs = self.roberta(
                input_ids=None,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=target_position_ids,
                head_mask=head_mask,
                inputs_embeds=step_embs,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

        
            sequence_output = outputs[0]
            logits = self.classifier(sequence_output)                
            span_probs = self.probs_of_pred(logits, pred_indexes)
            step_loss = torch.sum(span_probs)
            step_loss.backward()

            step_attribution = step_embs.grad.detach() * diff_embs / num_steps
            acc_attribution += step_attribution

            if step_i == 0:
                baseline_sum_logits = span_probs

        # sanity check
        # N_Layer * B * N_HEAD * L * L
        final_sum_logits = self.probs_of_pred(final_logits, pred_indexes)

        diff_sum_logits = final_sum_logits - baseline_sum_logits
        sum_attribution = torch.sum(acc_attribution, (2,1))
        # print(baseline_sum_logits.view([-1]))
        # print(final_sum_logits.view([-1]))
        # print(diff_sum_logits.view([-1]))
        # print(sum_attribution.view([-1]))
        # # exit()
        # print(acc_attribution.size())
        acc_attribution = torch.sum(acc_attribution, 2)
        return acc_attribution
