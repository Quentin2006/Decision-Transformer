"""A small Decision Transformer for discrete actions.

The architecture follows Chen et al. (2021) and the official gym code. Each
timestep contributes the tokens (R_t, s_t, a_t). Every token gets a learned
timestep embedding (in place of positional embeddings), then LayerNorm. A
causal transformer runs over the 3K tokens, and the action a_t is predicted
from the output at the s_t token, so it sees R_t and s_t but not a_t.

The paper's Atari setup used discrete actions, so this model does too: an
embedding table with one extra padding index, and cross-entropy loss.

With use_rtg=False the return token is dropped, leaving the sequence
(s_t, a_t). That is the behavior-cloning baseline, built from the same code.
"""

import torch
import torch.nn as nn

from dt import MAX_EP_LEN


class DecisionTransformer(nn.Module):
    def __init__(
        self,
        state_dim,
        n_actions=2,
        K=20,
        hidden=128,
        n_layer=3,
        n_head=1,
        dropout=0.1,
        use_rtg=True,
    ):
        super().__init__()
        self.config = dict(
            state_dim=state_dim,
            n_actions=n_actions,
            K=K,
            hidden=hidden,
            n_layer=n_layer,
            n_head=n_head,
            dropout=dropout,
            use_rtg=use_rtg,
        )
        self.K = K
        self.use_rtg = use_rtg
        self.tokens_per_step = 3 if use_rtg else 2

        self.embed_timestep = nn.Embedding(MAX_EP_LEN + 1, hidden)
        self.embed_return = nn.Linear(1, hidden)
        self.embed_state = nn.Linear(state_dim, hidden)
        self.embed_action = nn.Embedding(n_actions + 1, hidden)  # +1 = pad
        self.embed_ln = nn.LayerNorm(hidden)
        self.drop = nn.Dropout(dropout)

        layer = nn.TransformerEncoderLayer(
            d_model=hidden,
            nhead=n_head,
            dim_feedforward=4 * hidden,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            layer, num_layers=n_layer, enable_nested_tensor=False
        )
        self.final_ln = nn.LayerNorm(hidden)
        self.predict_action = nn.Linear(hidden, n_actions)

        L = self.tokens_per_step * K
        causal = torch.triu(torch.ones(L, L, dtype=torch.bool), diagonal=1)
        self.register_buffer("causal_mask", causal, persistent=False)

    def forward(self, states, actions, rtg, timesteps, mask):
        """Return action logits B×K×n_actions, one per state token."""
        B, K, _ = states.shape
        t_emb = self.embed_timestep(timesteps)
        s_tok = self.embed_state(states) + t_emb
        a_tok = self.embed_action(actions) + t_emb
        if self.use_rtg:
            r_tok = self.embed_return(rtg) + t_emb
            tokens = torch.stack([r_tok, s_tok, a_tok], dim=2)
        else:
            tokens = torch.stack([s_tok, a_tok], dim=2)
        n = self.tokens_per_step
        x = tokens.reshape(B, n * K, -1)
        x = self.drop(self.embed_ln(x))

        # Attention mask (True = blocked): causal, plus no query may attend
        # to a padded key. A padded query still attends to itself, so no row
        # is fully blocked; a fully blocked row would make softmax give NaN.
        L = n * K
        pad = (mask == 0).repeat_interleave(n, dim=1)  # B×L
        eye = torch.eye(L, dtype=torch.bool, device=x.device)
        attn = self.causal_mask[:L, :L] | (pad[:, None, :] & ~eye)
        attn = attn.repeat_interleave(self.config["n_head"], dim=0)
        x = self.transformer(x, mask=attn)
        x = self.final_ln(x).reshape(B, K, n, -1)
        state_out = x[:, :, n - 2]  # the s_t token is second to last
        return self.predict_action(state_out)

    @torch.no_grad()
    def get_action(self, states, actions, rtg, timesteps):
        """Greedy action for the latest step of each sequence.

        Inputs are unpadded histories (B×t×...). The last action slot is a
        placeholder; the model cannot see it when predicting from s_t.
        """
        K = self.K
        states, actions = states[:, -K:], actions[:, -K:]
        rtg, timesteps = rtg[:, -K:], timesteps[:, -K:]
        B, t = actions.shape
        if t < K:
            p = K - t
            states = torch.cat([states.new_zeros(B, p, states.shape[2]), states], 1)
            actions = torch.cat(
                [actions.new_full((B, p), self.config["n_actions"]), actions], 1
            )
            rtg = torch.cat([rtg.new_zeros(B, p, 1), rtg], 1)
            timesteps = torch.cat([timesteps.new_zeros(B, p), timesteps], 1)
        mask = torch.zeros(B, K, device=states.device)
        mask[:, K - t :] = 1.0
        logits = self.forward(states, actions, rtg, timesteps, mask)
        return logits[:, -1].argmax(-1)
