#!/usr/bin/env python3
"""
liquid_rl_trainer.py — Entrenamiento de RL con Red Líquida (LNN),
auto-juego con sombra EMA (Polyak) y currículo por fases.

Arquitectura:
  • Backbone: Closed-form Continuous-time cell (CfC/LNN) con
    clamp estricto del estado oculto en [-5, 5].
  • Auto-juego: sombra con actualización EMA continua.
  • Currículo en 3 fases: Random → Shadow → Shadow + Mentor Minimax.
  • Checkpoint condicional: guarda sólo si WR ∈ [0.45, 0.75],
    ∥g∥ < 10.0 y entropía > 0.1 simultáneamente.

El código asume un entorno que expone la siguiente interfaz:
  - reset() → obs: np.ndarray
  - step(action: int) → (obs, reward, done, info)
  - legal_actions() → list[int]    (para Minimax)
  - clone() → Env                  (para Minimax)
  - resultado() → int              (opcional, para juegos de suma cero)
"""

from __future__ import annotations
import os
import sys
import time
import random
import math
import copy
import json
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical


# ═══════════════════════════════════════════════════════════════════════════════
#  CONFIGURACIÓN
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class TrainingConfig:
    # --- Currículo por fases ---
    phase0_games: int = 2000
    phase1_games: int = 6000
    # Fase 2 arranca en phase0_games + phase1_games

    phase0_lr: float = 1e-3
    phase0_entropy_coef: float = 0.05

    phase1_lr: float = 5e-4
    phase1_entropy_coef: float = 0.02

    phase2_lr: float = 1e-4
    phase2_entropy_coef: float = 0.5
    phase2_entropy_floor: float = 0.20
    phase2_explore_eps: float = 0.15  # epsilon-greedy permanente
    # Decaimiento lineal por partida en Fase 2
    phase2_entropy_decay_per_game: float = 0.0  # se computa automáticamente

    # --- Currículo continuo (blended Random→Minimax progresivo) ---
    curriculum_blend: bool = True      # True = 4 fases híbrido, False = 3 fases legacy
    curriculum_warmup: int = 500      # partidas 100% random al inicio (breve)

    # --- 4-fase híbrido (curriculum_blend=True) ---
    phase1_games_4: int = 3000        # shadow self-play puro
    phase2_games_4: int = 5000        # shadow + M6 depth 4-5 transición
    lr_init: float = 5e-4
    lr_final: float = 1e-4
    entropy_init_4: float = 0.3
    entropy_final_4: float = 0.1
    pity_init_4: float = 0.30
    pity_final_4: float = 0.05
    blend_window: int = 100           # games de mezcla entre fases

    # --- Auto-juego / EMA (Polyak) ---
    tau: float = 0.995

    # --- Mentor Stochastic Minimax ---
    mentor_pity_init: float = 0.50
    mentor_pity_floor: float = 0.05
    mentor_pity_decay_per_game: float = 0.0  # se computa en __post_init__

    # --- Optimización ---
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_grad_norm: float = 10.0
    weight_decay: float = 1e-5

    # --- Checkpoint saludable (condiciones simultáneas) ---
    checkpoint_dir: str = "checkpoints"
    healthy_wr_min: float = 0.45
    healthy_wr_max: float = 0.75
    healthy_grad_max: float = 10.0
    healthy_entropy_min: float = 0.1

    # --- Evaluación ---
    eval_interval: int = 100
    eval_episodes: int = 50

    # --- Red ---
    obs_dim: int = 9
    action_dim: int = 9
    hidden_dim: int = 300
    vocab_size: int = 128  # caracteres español (incluye pad=0, unk=1, bos=2, eos=3)

    # --- Multi-task (TTT + Lenguaje) ---
    text_every_n: int = 5  # cada N partidas, 1 batch de lenguaje
    text_batch_len: int = 256  # tokens por batch de lenguaje
    text_lm_lr: float = 5e-4  # learning rate para lenguaje

    # --- Fase 2: LM post-TTT con replay ---
    dt_lm: float = 0.3       # paso de tiempo lento para lenguaje (contexto largo)
    dt_ttt: float = 1.0      # paso de tiempo rápido para TTT (reacción inmediata)
    lm_alpha: float = 0.5    # escala del loss LM cuando se combina con REINFORCE
    replay_buffer_size: int = 32   # episodios TTT retenidos en el buffer
    replay_interval: int = 10      # cada N batches LM, jugar 1 partida TTT de replay

    # --- LoRA (Low-Rank Adaptation) ---
    lora_rank: int = 0           # rango LoRA para lenguaje (0=desactivado)

    # --- System 1/2 Adaptive Computation ---
    s12_enabled: bool = False    # activar cómputo adaptativo (ACT-style)
    s12_max_iters: int = 3       # máximo de iteraciones de refinamiento
    s12_lambda: float = 0.01     # penalización por esfuerzo extra

    # --- Self-play + Hardening automático ---
    hardening_pity: float = 0.30
    hardening_games: int = 500
    hardening_patience_evals: int = 10   # triggers after this many evals without improvement
    hardening_wr_delta: float = 0.02     # minimum WR improvement to reset patience

    # --- General ---
    seed: int = 42
    device: str = "cpu"

    def __post_init__(self):
        total_phase2_entropy = max(0.001, self.phase2_entropy_coef - self.phase2_entropy_floor)
        self.phase2_entropy_decay_per_game = total_phase2_entropy / 40000.0
        self.phase2_total = self.phase0_games + self.phase1_games + 490000  # default ~500k total


# ═══════════════════════════════════════════════════════════════════════════════
#  TELEMETRÍA PARA FASE 2
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class TelemetryEntry:
    step: int = 0
    loss_lang: float = 0.0
    effort_penalty_mean: float = 0.0
    avg_iters_used: float = 0.0
    win_rate_ttt: float = 0.0
    draw_rate_ttt: float = 0.0


class TrainingTelemetry:
    """Registro estructurado de telemetría durante Fase 2 para análisis (paper)."""
    def __init__(self):
        self.entries: list[TelemetryEntry] = []

    def log(self, step: int, loss_lang: float = 0.0,
            effort: float = 0.0, iters: float = 0.0,
            wr: float = 0.0, dr: float = 0.0):
        self.entries.append(TelemetryEntry(
            step=step, loss_lang=loss_lang,
            effort_penalty_mean=effort, avg_iters_used=iters,
            win_rate_ttt=wr, draw_rate_ttt=dr,
        ))

    def save(self, path: str = "telemetria_fase2.json"):
        data = [
            {
                "step": e.step,
                "loss_lang": e.loss_lang,
                "effort_penalty_mean": e.effort_penalty_mean,
                "avg_iters_used": e.avg_iters_used,
                "win_rate_ttt": e.win_rate_ttt,
                "draw_rate_ttt": e.draw_rate_ttt,
            }
            for e in self.entries
        ]
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        n = len(data)
        print(f"  ✓ Telemetría exportada: {path} ({n} registros)")


# ═══════════════════════════════════════════════════════════════════════════════
#  TOKENIZER CHARACTER-LEVEL PARA ESPAÑOL
# ═══════════════════════════════════════════════════════════════════════════════

class SpanishTokenizer:
    """Tokenizer character-level para español.
    Vocab: a-z + ñ + vocales acentuadas + puntuación común + especiales.
    """
    def __init__(self):
        chars = "abcdefghijklmnñopqrstuvwxyzáéíóúüABCÇDEFGHIJKLMNÑOPQRSTUVWXYZÁÉÍÓÚÜ0123456789 .,!?¡¿;:()[]'\"-@#$%&*/=+<>{}~«»\n\t"
        self._itos = ['<PAD>', '<UNK>', '<BOS>', '<EOS>'] + list(chars)
        self._stoi = {c: i for i, c in enumerate(self._itos)}
        self.pad = 0
        self.unk = 1
        self.bos = 2
        self.eos = 3
        self.vocab_size = len(self._itos)

    def encode(self, text: str, max_len: int = 0) -> list[int]:
        ids = [self.bos] + [self._stoi.get(c, self.unk) for c in text] + [self.eos]
        if max_len > 0 and len(ids) > max_len:
            ids = ids[:max_len - 1] + [self.eos]
        return ids

    def decode(self, ids: list[int]) -> str:
        return ''.join(self._itos[i] if 0 <= i < len(self._itos) else '?' for i in ids if i > self.eos)

    def encode_batch(self, texts: list[str], max_len: int) -> torch.Tensor:
        batch = []
        for t in texts:
            ids = self.encode(t, max_len)
            if len(ids) < max_len:
                ids = ids + [self.pad] * (max_len - len(ids))
            batch.append(ids[:max_len])
        return torch.tensor(batch, dtype=torch.long)


# ═══════════════════════════════════════════════════════════════════════════════
#  BACKBONE: CELDA LÍQUIDA (Closed-form Continuous-time)
# ═══════════════════════════════════════════════════════════════════════════════

class LiquidCell(nn.Module):
    """
    Celda recurrente continua de tiempo líquido (CfC).

    Aproxima la EDO  dh/dt = f(h, x)  mediante una forma cerrada que
    evita resolvedores numéricos. El estado oculto se clamp ea [-5, 5]
    para garantizar estabilidad del BPTT.

    Reference: Hasani et al., "Closed-form Continuous-time Neural Networks", 2021.
    """

    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.net_input = nn.Sequential(
            nn.Linear(input_size + hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
        )
        self.net_tau = nn.Sequential(
            nn.Linear(input_size + hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.gate = nn.Linear(input_size + hidden_size, hidden_size)

    def forward(self, x: torch.Tensor, hx: torch.Tensor, dt: float = 1.0) -> torch.Tensor:
        """
        x:  [*, input_size]
        hx: [*, hidden_size]
        dt: paso de tiempo real (segundos entre observaciones)
        Retorna: [*, hidden_size]

        Solución cerrada de la EDO  dh/dt = gate * (B - h) / tau:

            h(t + dt) = h(t) + (B - h(t)) * (1 - exp(-gate * dt / tau))
        """
        fused = torch.cat([x, hx], dim=-1)

        B = self.net_input(fused)
        tau = torch.sigmoid(self.net_tau(fused))
        gate = torch.sigmoid(self.gate(fused))

        ode_step = 1.0 - torch.exp(-gate * dt / (tau + 1e-6))
        h_new = hx + (B - hx) * ode_step

        h_new = torch.clamp(h_new, -5.0, 5.0)
        return h_new


class LoRALayer(nn.Module):
    """Low-Rank Adaptation (LoRA) para una proyección lineal.

    En lugar de modificar los pesos base, aprende matrices de bajo rango
    que producen un delta en la proyección:  h' = h + α/r · h·Aᵀ·Bᵀ

    Args:
        in_features: dimensión de entrada
        out_features: dimensión de salida (normalmente igual a in_features)
        rank: rango de la descomposición (8-16 típico)
        alpha: factor de escalado (1.0 por defecto)
    """
    def __init__(self, in_features: int, out_features: int, rank: int = 8, alpha: float = 1.0):
        super().__init__()
        self.rank = rank
        self.scaling = alpha / max(rank, 1)
        self.lora_A = nn.Parameter(torch.randn(rank, in_features) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x @ self.lora_A.T @ self.lora_B.T) * self.scaling


class LiquidAgent(nn.Module):
    """
    Agente completo: codificación → LNN → cabezas de política y valor.

    La entrada se normaliza explícitamente antes del backbone.

    Soporta LoRA para lenguaje: cuando lora_rank > 0, el backbone CfC
    se congela permanentemente y el lenguaje se aprende mediante
    adaptadores de bajo rango en la proyección del embedding de texto.
    """

    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 256, vocab_size: int = 128, lora_rank: int = 0):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.lora_rank = lora_rank

        self.register_buffer("obs_mean", torch.zeros(obs_dim))
        self.register_buffer("obs_std", torch.ones(obs_dim))

        self.liquid = LiquidCell(obs_dim, hidden_dim)
        self.policy_head = nn.Linear(hidden_dim, action_dim)
        self.value_head = nn.Linear(hidden_dim, 1)
        self.text_embed = nn.Embedding(vocab_size, obs_dim)
        self.lang_norm = nn.LayerNorm(hidden_dim)
        self.lang_head = nn.Linear(hidden_dim, vocab_size)
        self.lang_drop = nn.Dropout(0.1)

        # --- LoRA: adaptador de bajo rango para lenguaje ---
        if lora_rank > 0:
            self.lora_proj = LoRALayer(obs_dim, obs_dim, rank=lora_rank, alpha=1.0)
        else:
            self.lora_proj = None

        # --- System 1/2 Adaptive Computation ---
        self.effort_gate = nn.Linear(hidden_dim, 1)
        nn.init.constant_(self.effort_gate.bias, 2.0)  # sesgo a parar pronto
        self.s12_enabled: bool = False
        self.s12_max_iters: int = 3

    def forward_step(
        self, obs: torch.Tensor, hx: torch.Tensor, dt: float = 1.0
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        obs_norm = (obs - self.obs_mean) / (self.obs_std + 1e-8)
        h_new = self.liquid(obs_norm, hx, dt=dt)
        logits = self.policy_head(h_new)
        value = self.value_head(h_new).squeeze(-1)
        return logits, value, h_new

    def forward_adaptive(
        self, obs: torch.Tensor, hx: torch.Tensor, dt: float = 1.0, max_iters: int | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]:
        """
        System 1/2 con soft accumulation (ACT-style).
        En inferencia (torch.no_grad) el bucle se corta temprano si hay confianza.
        En training el gradiente fluye por todas las iteraciones via pesos ponderados.

        Retorna: (logits, value, hx_new, effort_penalty, n_iters)
        """
        max_iters = max_iters if max_iters is not None else self.s12_max_iters
        batch_size = hx.size(0)
        device = hx.device

        h_actual = hx
        acum_logits = None
        acum_value = None
        remain = torch.ones(batch_size, 1, device=device)
        effort_penalty = torch.zeros(batch_size, 1, device=device)

        for i in range(max_iters):
            logits_i, value_i, h_nuevo = self.forward_step(obs, h_actual, dt=dt)

            if acum_logits is None:
                acum_logits = torch.zeros_like(logits_i)
                acum_value = torch.zeros_like(value_i)

            p_halting = torch.sigmoid(self.effort_gate(h_actual))
            if i == max_iters - 1:
                p_halting = remain

            peso = torch.min(p_halting, remain)
            remain = (remain - peso).clamp(min=0.0)

            acum_logits = acum_logits + peso * logits_i
            acum_value = acum_value + peso * value_i
            effort_penalty = effort_penalty + peso * (i + 1)

            h_actual = h_nuevo

            if torch.all(remain <= 0.0) and i < max_iters - 1:
                break

        return acum_logits, acum_value, h_actual, effort_penalty, i + 1

    def get_action(
        self,
        obs: np.ndarray,
        hx: torch.Tensor,
        legal_mask: torch.Tensor | None = None,
        deterministic: bool = False,
        dt: float = 1.0,
    ) -> tuple[int, torch.Tensor]:
        obs_t = torch.from_numpy(obs).float().to(hx.device)
        if obs_t.dim() == 1:
            obs_t = obs_t.unsqueeze(0)
        with torch.no_grad():
            if self.s12_enabled:
                logits, value, hx_new, _, _ = self.forward_adaptive(obs_t, hx, dt=dt)
            else:
                logits, value, hx_new = self.forward_step(obs_t, hx, dt=dt)
            if legal_mask is not None:
                logits = logits.masked_fill(~legal_mask, -float("inf"))
            if deterministic:
                action = logits.argmax(dim=-1).item()
            else:
                dist = Categorical(logits=logits)
                action = dist.sample().item()
        return action, hx_new

    def forward_lm_step(
        self, token_id: torch.Tensor, hx: torch.Tensor, dt: float = 0.3
    ) -> tuple[torch.Tensor, torch.Tensor]:
        emb = self.text_embed(token_id)
        if emb.dim() == 3:
            emb = emb.squeeze(1)
        # LoRA: modifica la proyección del embedding sin tocar el backbone
        if self.lora_proj is not None:
            emb = emb + self.lora_proj(emb)
        h_new = self.liquid(emb, hx, dt=dt)
        h_dropped = self.lang_drop(self.lang_norm(h_new))
        logits = self.lang_head(h_dropped)
        return logits, h_new

    def forward_lm_adaptive(
        self, token_id: torch.Tensor, hx: torch.Tensor, dt: float = 0.3, max_iters: int | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        """
        System 1/2 para lenguaje con soft accumulation.
        Retorna: (logits, hx_new, effort_penalty, n_iters)
        """
        max_iters = max_iters if max_iters is not None else self.s12_max_iters
        batch_size = hx.size(0)
        device = hx.device

        h_actual = hx
        acum_logits = None
        remain = torch.ones(batch_size, 1, device=device)
        effort_penalty = torch.zeros(batch_size, 1, device=device)

        for i in range(max_iters):
            logits_i, h_nuevo = self.forward_lm_step(token_id, h_actual, dt=dt)

            if acum_logits is None:
                acum_logits = torch.zeros_like(logits_i)

            p_halting = torch.sigmoid(self.effort_gate(h_actual))
            if i == max_iters - 1:
                p_halting = remain

            peso = torch.min(p_halting, remain)
            remain = (remain - peso).clamp(min=0.0)

            acum_logits = acum_logits + peso * logits_i
            effort_penalty = effort_penalty + peso * (i + 1)

            h_actual = h_nuevo

            if torch.all(remain <= 0.0) and i < max_iters - 1:
                break

        return acum_logits, h_actual, effort_penalty, i + 1

    def generate(self, prompt_ids: list[int], hx: torch.Tensor, max_len: int = 100, temp: float = 1.0, dt: float = 0.3) -> tuple[list[int], torch.Tensor]:
        self.eval()
        generated = list(prompt_ids)
        for _ in range(max_len):
            inp = torch.tensor([generated[-1]], device=hx.device)
            if self.s12_enabled:
                logits, hx, _, _ = self.forward_lm_adaptive(inp, hx, dt=dt)
            else:
                logits, hx = self.forward_lm_step(inp, hx, dt=dt)
            logits = logits.squeeze(0) / temp
            probs = torch.softmax(logits, dim=-1)
            if temp < 0.1:
                next_id = logits.argmax(dim=-1).item()
            else:
                next_id = torch.multinomial(probs, 1).item()
            generated.append(next_id)
            if next_id == 3:  # <EOS>
                break
        return generated, hx


# ═══════════════════════════════════════════════════════════════════════════════
#  AGENTE SOMBRA (EMA / Polyak)
# ═══════════════════════════════════════════════════════════════════════════════

class ShadowAgent:
    """
    Copia congelada del agente con actualización EMA continua.

        θ_s ← τ · θ_s + (1 - τ) · θ_a

    τ ∈ [0.995, 0.999] mantiene el entorno de auto-juego casi-estacionario.
    """

    def __init__(self, agent: nn.Module, tau: float):
        self.tau = tau
        self.model = copy.deepcopy(agent)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False

    def update(self, agent: nn.Module):
        with torch.no_grad():
            for p_s, p_a in zip(self.model.parameters(), agent.parameters()):
                p_s.data.copy_(self.tau * p_s.data + (1.0 - self.tau) * p_a.data)

    def get_action(
        self, obs: np.ndarray, hx: torch.Tensor, legal_mask: torch.Tensor | None = None
    ) -> tuple[int, torch.Tensor]:
        return self.model.get_action(obs, hx, legal_mask=legal_mask, deterministic=True)


# ═══════════════════════════════════════════════════════════════════════════════
#  AGENTE ALEATORIO
# ═══════════════════════════════════════════════════════════════════════════════

class RandomAgent:
    """Política uniforme sobre acciones legales."""

    def get_action(self, legal: list[int]) -> int:
        return random.choice(legal) if legal else 0

    def reset(self):
        pass


# ═══════════════════════════════════════════════════════════════════════════════
#  MENTOR STOCHASTIC MINIMAX
# ═══════════════════════════════════════════════════════════════════════════════

class StochasticMinimax:
    """
    Oponente Minimax con ruido epsilon-greedy (pity).

    Juega la jugada Minimax a profundidad `depth` con probabilidad (1-pity)*(1-noise_prob),
    pero ocasionalmente hace jugadas aleatorias.

    depth=6 permite que el agente gane algunas partidas (señal positiva).
    depth=9 es invencible (solo empates).
    """

    def __init__(self, mentor_player: int = -1, depth: int = 6, noise_prob: float = 0.05):
        self.mentor_player = mentor_player
        self.depth = depth
        self.noise_prob = noise_prob

    def get_action(self, env: Any, legal: list[int], pity: float = 0.0, depth: int | None = None) -> int:
        if not legal:
            return -1
        if random.random() < max(pity, self.noise_prob):
            return random.choice(legal)
        return self._best_action(env, legal, depth or self.depth)

    def _best_action(self, env: Any, legal: list[int], depth: int) -> int:
        best_a = legal[0]
        best_val = -float("inf")
        for a in legal:
            c = env.clone()
            c.step(a, player=env.HUMANO)
            val = self._minimax(c, depth, -float("inf"), float("inf"), False)
            if val > best_val:
                best_val = val
                best_a = a
        return best_a

    def _minimax(
        self, env: Any, depth: int, alpha: float, beta: float, maximizing: bool
    ) -> float:
        done = env.terminado() if hasattr(env, "terminado") else env.done
        legal = env.legal_actions() if hasattr(env, "legal_actions") else []

        if done or depth == 0 or not legal:
            r = env.resultado() if hasattr(env, "resultado") else 0.0
            return r * self.mentor_player

        if maximizing:
            val = -float("inf")
            for a in legal:
                c = env.clone()
                c.step(a, player=env.HUMANO)
                val = max(val, self._minimax(c, depth - 1, alpha, beta, False))
                alpha = max(alpha, val)
                if alpha >= beta:
                    break
            return val
        else:
            val = float("inf")
            for a in legal:
                c = env.clone()
                c.step(a, player=env.AGENTE)
                val = min(val, self._minimax(c, depth - 1, alpha, beta, True))
                beta = min(beta, val)
                if alpha >= beta:
                    break
            return val

    def evaluate(self, env: Any) -> float:
        """Evalúa la posición desde la perspectiva del mentor (oponente).
        Retorna > 0 si el oponente tiene ventaja, < 0 si el agente, ≈ 0 si tablas.
        """
        legal = env.legal_actions() if hasattr(env, "legal_actions") else []
        done = env.terminado() if hasattr(env, "terminado") else getattr(env, "done", False)
        if done or not legal:
            r = env.resultado() if hasattr(env, "resultado") else 0.0
            return r * self.mentor_player
        return self._minimax(env, self.depth, -float("inf"), float("inf"), True)


# ═══════════════════════════════════════════════════════════════════════════════
#  GESTOR DE CURRÍCULO
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class PhaseInfo:
    id: int
    name: str
    lr: float
    entropy_coef: float
    use_shadow: bool
    use_mentor: bool
    opponent: str


class CurriculumManager:
    """
    Gestor de currículo con dos modos:

    curriculum_blend=False (legacy): 3 fases discretas (Random → Shadow → Mentor M6).
    curriculum_blend=True (nuevo): 4 fases híbridas con transición suave:
      F0: Random puro (warmup)
      F1: Shadow self-play puro
      F2: Shadow + Mentor M6 depth 4-5 (alternado 50/50)
      F3: Shadow + Mentor M6 depth 6 (alternado 50/50) + pity decayente

    Hardening: si WR vs M6 no mejora, se inyectan K partidas vs M6 con pity fijo.
    """

    def __init__(self, config: TrainingConfig):
        self.cfg = config
        # Legacy 3-phase boundaries
        self.phase0_start = 0
        self.phase1_start = config.phase0_games
        self.phase2_start = config.phase0_games + config.phase1_games
        # 4-phase boundaries (computados)
        self._w = config.curriculum_warmup
        self._p1_end = self._w + config.phase1_games_4
        self._p2_end = self._p1_end + config.phase2_games_4
        # Hardening state
        self.hardening_active = False
        self.hardening_remaining = 0
        # Blend: última oposición real usada para transición suave
        self._last_was_shadow = True

    def activate_hardening(self):
        self.hardening_active = True
        self.hardening_remaining = self.cfg.hardening_games

    def tick_hardening(self) -> bool:
        if not self.hardening_active:
            return False
        self.hardening_remaining -= 1
        if self.hardening_remaining <= 0:
            self.hardening_active = False
            return True
        return False

    def _progress(self, game: int, num_games: int) -> float:
        """Progreso global 0.0→1.0 después del warmup."""
        if game <= self._w:
            return 0.0
        denom = max(1, num_games - self._w)
        return min(1.0, (game - self._w) / denom)

    def get_phase(self, game: int) -> int:
        if not self.cfg.curriculum_blend:
            if game < self.phase1_start:
                return 0
            elif game < self.phase2_start:
                return 1
            return 2
        # 4-phase
        if game <= self._w:
            return 0
        elif game <= self._p1_end:
            return 1
        elif game <= self._p2_end:
            return 2
        return 3

    def _in_blend(self, game: int) -> int | None:
        """Retorna la fase DESTINO si estamos en ventana de blend, None en otro caso."""
        if not self.cfg.curriculum_blend:
            return None
        bw = self.cfg.blend_window
        if self._w < game <= self._w + bw:
            return 1
        if self._p1_end < game <= self._p1_end + bw:
            return 2
        if self._p2_end < game <= self._p2_end + bw:
            return 3
        return None

    def _blend_ratio(self, game: int, boundary: int) -> float:
        """0.0 en boundary, 1.0 en boundary+blend_window."""
        bw = self.cfg.blend_window
        return min(1.0, max(0.0, (game - boundary) / max(1, bw)))

    def get_phase_info(self, game: int) -> PhaseInfo:
        if not self.cfg.curriculum_blend:
            phase = self.get_phase(game)
            if phase == 0:
                return PhaseInfo(id=0, name="WARM-UP (Random)",
                    lr=self.cfg.phase0_lr,
                    entropy_coef=self.cfg.phase0_entropy_coef,
                    use_shadow=False, use_mentor=False, opponent="Random")
            elif phase == 1:
                return PhaseInfo(id=1, name="DESARROLLO (Shadow)",
                    lr=self.cfg.phase1_lr,
                    entropy_coef=self.cfg.phase1_entropy_coef,
                    use_shadow=True, use_mentor=False, opponent="Shadow")
            else:
                return PhaseInfo(id=2, name="COMPETICIÓN (Stochastic Minimax)",
                    lr=self.cfg.phase2_lr,
                    entropy_coef=self.cfg.phase2_entropy_coef,
                    use_shadow=False, use_mentor=True, opponent="StochasticMinimax")
        if self.hardening_active:
            return PhaseInfo(id=4, name="HARDENING vs M6",
                lr=self.get_lr(game),
                entropy_coef=self.get_entropy_coef(game),
                use_shadow=False, use_mentor=True, opponent="StochasticMinimax")
        phase = self.get_phase(game)
        lr = self.get_lr(game)
        ec = self.get_entropy_coef(game)
        if phase == 0:
            return PhaseInfo(0, "WARM-UP (Random)", lr, ec, False, False, "Random")
        elif phase == 1:
            return PhaseInfo(1, "SELF-PLAY (Shadow)", lr, ec, True, False, "Shadow")
        elif phase == 2:
            return PhaseInfo(2, "TRANSICIÓN (Shadow+M6 D4-5)", lr, ec, True, True,
                            "Alternating(Shadow|MentorD4-5)")
        else:
            return PhaseInfo(3, "COMPETICIÓN (Shadow+M6 D6)", lr, ec, True, True,
                            "Alternating(Shadow|MentorD6)")

    def get_entropy_coef(self, game: int) -> float:
        if not self.cfg.curriculum_blend:
            if self.get_phase(game) < 2:
                return self.get_phase_info(game).entropy_coef
            games_in_phase2 = max(0, game - self.phase2_start)
            decayed = self.cfg.phase2_entropy_coef - self.cfg.phase2_entropy_decay_per_game * games_in_phase2
            return max(self.cfg.phase2_entropy_floor, decayed)
        # 4-phase: lineal 0.3 → 0.1 sobre ~100k games
        p = self._progress(game, 100000)
        return max(self.cfg.entropy_final_4,
                   self.cfg.entropy_init_4 * (1.0 - 0.67 * min(1.0, p)))

    def get_mentor_pity(self, game: int, num_games: int) -> float:
        if not self.cfg.curriculum_blend:
            if self.get_phase(game) < 2:
                return 0.0
            games_in_phase2 = max(0, game - self.phase2_start)
            total_phase2 = max(1, num_games - self.phase2_start)
            fraction = games_in_phase2 / total_phase2
            return max(self.cfg.mentor_pity_floor, self.cfg.mentor_pity_init * (1.0 - fraction))
        if self.hardening_active:
            return self.cfg.hardening_pity
        phase = self.get_phase(game)
        if phase < 3:
            return 0.0
        # Fase 3: pity 0.30 → 0.05 lineal
        games_in = max(0, game - self._p2_end)
        remaining = max(1, num_games - self._p2_end)
        p = games_in / remaining
        return max(self.cfg.pity_final_4, self.cfg.pity_init_4 * (1.0 - p))

    def get_mentor_prob(self, game: int, num_games: int) -> float:
        """mentor_prob=1.0 en Fases 2 y 3 (siempre mentor, alternado con shadow)."""
        if not self.cfg.curriculum_blend:
            return 1.0 if self.get_phase(game) >= 2 else 0.0
        if self.hardening_active:
            return 1.0
        return 0.0 if self.get_phase(game) < 2 else 1.0

    def get_mentor_depth(self, game: int, num_games: int) -> int:
        if not self.cfg.curriculum_blend:
            return 6
        phase = self.get_phase(game)
        if phase >= 3:
            return 6
        if phase == 2:
            # Depth progresivo 4→5 a lo largo de la fase
            p = (game - self._p1_end) / max(1, self.cfg.phase2_games_4)
            return 4 if random.random() > p else 5
        return 0  # no mentor

    def get_lr(self, game: int) -> float:
        if not self.cfg.curriculum_blend:
            return self.get_phase_info(game).lr
        # 4-phase: lineal 5e-4 → 1e-4 sobre ~100k games
        p = self._progress(game, 100000)
        return max(self.cfg.lr_final, self.cfg.lr_init * (1.0 - 0.8 * min(1.0, p)))

    def select_opponent(self, game: int, num_games: int) -> dict:
        """Retorna configuración del oponente para esta partida,
        incluyendo manejo de blend windows."""
        if not self.cfg.curriculum_blend:
            info = self.get_phase_info(game)
            return dict(use_shadow=info.use_shadow, use_mentor=info.use_mentor,
                        opp_name=info.opponent, opp_depth=self.get_mentor_depth(game, num_games),
                        opp_pity=self.get_mentor_pity(game, num_games))

        # --- Hardening ---
        if self.hardening_active:
            return dict(use_shadow=False, use_mentor=True, opp_name="M6[HARD]",
                        opp_depth=6, opp_pity=self.cfg.hardening_pity)

        phase = self.get_phase(game)
        pity = self.get_mentor_pity(game, num_games)

        # --- Blend entre fases ---
        blend_target = self._in_blend(game)
        if blend_target is not None:
            # Elegir entre fase actual y destino según blend_ratio
            if blend_target == 1:
                # 0→1: Random vs Shadow
                ratio = self._blend_ratio(game, self._w)
                if random.random() < ratio:
                    return dict(use_shadow=True, use_mentor=False, opp_name="SHADOW",
                                opp_depth=0, opp_pity=0.0)
                return dict(use_shadow=False, use_mentor=False, opp_name="RANDOM",
                            opp_depth=0, opp_pity=0.0)
            elif blend_target == 2:
                # 1→2: Shadow puro vs Shadow+M6
                ratio = self._blend_ratio(game, self._p1_end)
                if random.random() < ratio:
                    depth = 4 if random.random() > 0.5 else 5
                    use_m = random.random() < 0.5
                    return dict(use_shadow=not use_m, use_mentor=use_m,
                                opp_name=f"M{depth}" if use_m else "SHADOW",
                                opp_depth=depth, opp_pity=0.0)
                return dict(use_shadow=True, use_mentor=False, opp_name="SHADOW",
                            opp_depth=0, opp_pity=0.0)
            else:  # 2→3
                ratio = self._blend_ratio(game, self._p2_end)
                if random.random() < ratio:
                    use_m = random.random() < 0.5
                    return dict(use_shadow=not use_m, use_mentor=use_m,
                                opp_name="M6" if use_m else "SHADOW",
                                opp_depth=6, opp_pity=max(0.05, pity * ratio))
                depth = 4 if random.random() > 0.5 else 5
                use_m = random.random() < 0.5
                return dict(use_shadow=not use_m, use_mentor=use_m,
                            opp_name=f"M{depth}" if use_m else "SHADOW",
                            opp_depth=depth, opp_pity=0.0)

        # --- Fase normal (sin blend) ---
        if phase == 0:
            return dict(use_shadow=False, use_mentor=False, opp_name="RANDOM",
                        opp_depth=0, opp_pity=0.0)
        if phase == 1:
            return dict(use_shadow=True, use_mentor=False, opp_name="SHADOW",
                        opp_depth=0, opp_pity=0.0)
        # Fases 2 y 3: alternar 50/50 shadow vs mentor
        depth = self.get_mentor_depth(game, num_games)
        use_m = random.random() < 0.5
        return dict(use_shadow=not use_m, use_mentor=use_m,
                    opp_name=f"M{depth}" if use_m else "SHADOW",
                    opp_depth=depth, opp_pity=pity if use_m else 0.0)


# ═══════════════════════════════════════════════════════════════════════════════
#  HEALTH CHECKER (Checkpoint condicional)
# ═══════════════════════════════════════════════════════════════════════════════

class HealthChecker:
    """
    Checkpoint condicional — guarda healthy_checkpoint.pth sólo si se
    cumplen TODAS las condiciones simultáneamente:
      1. Win Rate contra sombra ∈ [WR_min, WR_max]
      2. ∥gradiente∥ < grad_max
      3. Entropía de política > entropy_min

    best_checkpoint.pth es independiente del techo de WR (0.75):
    se actualiza siempre que WR supere al mejor anterior, manteniendo
    norma de gradiente y entropía saludables.
    """

    def __init__(self, agent: nn.Module, config: TrainingConfig):
        self.agent = agent
        self.cfg = config
        self.best_wr = 0.0
        os.makedirs(config.checkpoint_dir, exist_ok=True)
        self._ckpt_path = os.path.join(config.checkpoint_dir, "healthy_checkpoint.pth")
        self._best_path = os.path.join(config.checkpoint_dir, "best_checkpoint.pth")

    def is_healthy(self, wr: float, grad_norm: float, entropy: float) -> bool:
        return (
            self.cfg.healthy_wr_min <= wr <= self.cfg.healthy_wr_max
            and grad_norm < self.cfg.healthy_grad_max
            and entropy > self.cfg.healthy_entropy_min
        )

    def save_if_healthy(self, wr: float, grad_norm: float, entropy: float) -> bool:
        saved = False

        # healthy_checkpoint: AND estricto (fusible anti-colapso)
        if self.is_healthy(wr, grad_norm, entropy):
            torch.save(self.agent.state_dict(), self._ckpt_path)
            saved = True

        # best_checkpoint: independiente del techo de WR,
        # sólo requiere gradiente y entropía saludables
        if grad_norm < self.cfg.healthy_grad_max and entropy > self.cfg.healthy_entropy_min:
            if wr > self.best_wr:
                self.best_wr = wr
                torch.save(self.agent.state_dict(), self._best_path)

        return saved

    def has_checkpoint(self) -> bool:
        return os.path.exists(self._ckpt_path)

    def load_healthy(self) -> bool:
        if os.path.exists(self._ckpt_path):
            sd = torch.load(self._ckpt_path, map_location="cpu")
            self.agent.load_state_dict(sd)
            return True
        return False

    def get_checkpoint_path(self) -> str:
        return self._ckpt_path

    def get_best_path(self) -> str:
        return self._best_path


# ═══════════════════════════════════════════════════════════════════════════════
#  FUNCIÓN DE PÉRDIDA (REINFORCE con línea base)
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class EpisodeBuffer:
    obs: list[np.ndarray] = field(default_factory=list)
    actions: list[int] = field(default_factory=list)
    rewards: list[float] = field(default_factory=list)
    legal: list[list[int]] = field(default_factory=list)

    @property
    def length(self) -> int:
        return len(self.obs)


class ReplayBuffer:
    """Buffer circular de episodios TTT para replay en Fase 2."""
    def __init__(self, maxlen: int = 32):
        self.buffer: deque[EpisodeBuffer] = deque(maxlen=maxlen)

    def add(self, episode: EpisodeBuffer):
        self.buffer.append(episode)

    def sample(self) -> list[EpisodeBuffer]:
        return list(self.buffer)

    def __len__(self) -> int:
        return len(self.buffer)


def compute_reinforce_loss(
    agent: LiquidAgent,
    buffer: EpisodeBuffer,
    gamma: float,
    entropy_coef: float,
    device: torch.device,
    s12_lambda: float = 0.0,
    gae_lambda: float = 0.0,
) -> dict[str, torch.Tensor]:
    T = buffer.length
    if T == 0:
        return {
            "total": torch.tensor(0.0, device=device),
            "policy": torch.tensor(0.0, device=device),
            "value": torch.tensor(0.0, device=device),
            "entropy": torch.tensor(0.0, device=device),
            "effort": torch.tensor(0.0, device=device),
            "n_iters": 0,
        }

    hx = torch.zeros(1, agent.hidden_dim, device=device)
    all_log_probs: list[torch.Tensor] = []
    all_entropies: list[torch.Tensor] = []
    all_values: list[torch.Tensor] = []
    total_effort = torch.zeros(1, 1, device=device)
    total_n_iters = 0

    for t in range(T):
        obs_t = torch.from_numpy(buffer.obs[t]).float().to(device).unsqueeze(0)
        obs_norm = (obs_t - agent.obs_mean) / (agent.obs_std + 1e-8)

        if agent.s12_enabled:
            logits, value, hx, effort, n_iters = agent.forward_adaptive(obs_norm, hx)
            total_effort = total_effort + effort
            total_n_iters += n_iters
        else:
            logits, value, hx = agent.forward_step(obs_norm, hx)

        # ── Fix 1: enmascarar acciones ilegales ANTES de Categorical ──
        legal_mask = torch.zeros(agent.action_dim, dtype=torch.bool, device=device)
        for a in buffer.legal[t]:
            legal_mask[a] = True
        logits = logits.masked_fill(~legal_mask, -float("inf"))

        action_t = torch.tensor([buffer.actions[t]], device=device)
        dist = Categorical(logits=logits)
        log_prob = dist.log_prob(action_t)
        entropy = dist.entropy()

        all_log_probs.append(log_prob)
        all_entropies.append(entropy)
        all_values.append(value)

    log_probs = torch.stack(all_log_probs)
    entropies = torch.stack(all_entropies)
    values = torch.stack(all_values).squeeze()

    # ── GAE(λ) para advantage más preciso — credit assignment fino ──
    rewards_t = torch.tensor(buffer.rewards, device=device, dtype=torch.float32)
    with torch.no_grad():
        gae = torch.zeros(T, device=device)
        last_gae = 0.0
        for t in reversed(range(T)):
            delta = rewards_t[t] + gamma * (values[t + 1] if t + 1 < T else 0.0) - values[t]
            last_gae = delta + gamma * gae_lambda * last_gae
            gae[t] = last_gae
        returns = gae + values.detach()

    advantage = gae

    policy_loss = -(log_probs * advantage).mean()
    value_loss = F.mse_loss(values, returns.detach())
    entropy_loss = -entropy_coef * entropies.mean()

    # Entropy floor duro: penaliza si H < 0.2 para evitar colapso total
    h_mean = entropies.mean()
    if h_mean < 0.20:
        entropy_loss = entropy_loss - 1.0 * (0.20 - h_mean)

    total_loss = policy_loss + value_loss + entropy_loss

    if agent.s12_enabled and s12_lambda > 0.0:
        total_loss = total_loss + s12_lambda * total_effort.mean()

    return {
        "total": total_loss,
        "policy": policy_loss.detach(),
        "value": value_loss.detach(),
        "entropy": entropies.mean().detach(),
        "effort": total_effort.mean().detach(),
        "n_iters": total_n_iters,
    }


def compute_language_loss(
    agent: LiquidAgent,
    token_ids: torch.Tensor,
    device: torch.device,
    dt: float = 0.3,
    s12_lambda: float = 0.0,
) -> torch.Tensor:
    """Cross-entropy loss de lenguaje: predice el siguiente carácter.

    Args:
        token_ids: (B, L) — batch de secuencias tokenizadas (con <BOS> al inicio).
        dt: paso de tiempo para la CfC (dt_lm=0.3 para contexto largo).
        s12_lambda: coeficiente de penalización por esfuerzo (System 1/2).
    """
    B, L = token_ids.shape
    hx = torch.zeros(B, agent.hidden_dim, device=device)
    total_loss = 0.0
    total_effort = torch.zeros(B, 1, device=device)
    count = 0
    for t in range(L - 1):
        inp = token_ids[:, t]
        if agent.s12_enabled:
            logits, hx, effort, _ = agent.forward_lm_adaptive(inp, hx, dt=dt)
            total_effort = total_effort + effort
        else:
            logits, hx = agent.forward_lm_step(inp, hx, dt=dt)
        target = token_ids[:, t + 1]
        loss = F.cross_entropy(logits, target, ignore_index=0)  # ignore <PAD>
        total_loss = total_loss + loss
        count += 1
    avg_loss = total_loss / count
    avg_effort = (total_effort / count).mean().item() if agent.s12_enabled else 0.0
    if agent.s12_enabled and s12_lambda > 0.0:
        avg_loss = avg_loss + s12_lambda * (total_effort / count).mean()
    return avg_loss, avg_effort


# ═══════════════════════════════════════════════════════════════════════════════
#  ENTORNO ADAPTADOR PARA TIC-TAC-TOE
# ═══════════════════════════════════════════════════════════════════════════════

class TicTacToeEnv:
    """
    Envoltorio Gym-like para el Tablero TTT.

    step(action: int) → (obs, reward, done, info)
    La observación es el tablero aplanado: 9 floats en {0, 1, 2}.
    El agente juega como 'AGENTE=2', el oponente como 'HUMANO=1'.
    """

    VACIO = 0
    HUMANO = 1
    AGENTE = 2

    def __init__(self):
        self.grid = [[self.VACIO] * 3 for _ in range(3)]
        self.done = False
        self._winner = self.VACIO

    def reset(self) -> np.ndarray:
        self.grid = [[self.VACIO] * 3 for _ in range(3)]
        self.done = False
        self._winner = self.VACIO
        return self._obs()

    def step(self, action: int, player: int | None = None) -> tuple[np.ndarray, float, bool, dict]:
        if player is None:
            player = self.AGENTE
        r, c = divmod(action, 3)
        if self.grid[r][c] != self.VACIO:
            return self._obs(), -1.0, True, {"valid": False}
        self.grid[r][c] = player
        self._winner = self._check_winner()
        self.done = self._winner != self.VACIO or not any(
            self.VACIO in row for row in self.grid
        )
        reward = 0.0 if not self.done else self.resultado()
        return self._obs(), reward, self.done, {"valid": True}

    def legal_actions(self) -> list[int]:
        return [r * 3 + c for r in range(3) for c in range(3)
                if self.grid[r][c] == self.VACIO]

    def clone(self) -> TicTacToeEnv:
        c = TicTacToeEnv()
        c.grid = [row[:] for row in self.grid]
        c.done = self.done
        c._winner = self._winner
        return c

    def resultado(self) -> int:
        if self._winner == self.AGENTE:
            return 1
        if self._winner == self.HUMANO:
            return -1
        return 0

    def terminado(self) -> bool:
        return self.done

    def _obs(self) -> np.ndarray:
        return np.array(self.grid, dtype=np.float32).flatten()

    def _check_winner(self) -> int:
        g = self.grid
        for i in range(3):
            if g[i][0] == g[i][1] == g[i][2] != self.VACIO:
                return g[i][0]
            if g[0][i] == g[1][i] == g[2][i] != self.VACIO:
                return g[0][i]
        if g[0][0] == g[1][1] == g[2][2] != self.VACIO:
            return g[0][0]
        if g[0][2] == g[1][1] == g[2][0] != self.VACIO:
            return g[0][2]
        return self.VACIO

    def render(self) -> str:
        sym = {self.VACIO: ".", self.HUMANO: "X", self.AGENTE: "O"}
        return "\n".join(" ".join(sym[c] for c in row) for row in self.grid)


# ═══════════════════════════════════════════════════════════════════════════════
#  MÓDULO DE AUTO-JUEGO
# ═══════════════════════════════════════════════════════════════════════════════

class SelfPlayModule:
    """
    Gestiona una partida entre el agente y un oponente (Random, Shadow o Mentor).

    Retorna un EpisodeBuffer con la trayectoria completa del agente.
    """

    def __init__(
        self,
        agent: LiquidAgent,
        shadow: ShadowAgent | None,
        mentor: StochasticMinimax | None,
        env: Any,
        device: torch.device,
        phase: int,
    ):
        self.agent = agent
        self.shadow = shadow
        self.mentor = mentor
        self.env = env
        self.device = device
        self.phase = phase

    def play_episode(
        self, use_mentor: bool = False, pity: float = 0.0, mentor: StochasticMinimax | None = None,
        mentor_depth: int | None = None, explore_eps: float = 0.0,
    ) -> EpisodeBuffer:
        buffer = EpisodeBuffer()
        hx_agent = torch.zeros(1, self.agent.hidden_dim, device=self.device)
        hx_opp = torch.zeros(1, self.agent.hidden_dim, device=self.device)

        obs = self.env.reset()
        done = False
        turn = 0

        while not done:
            legal = self.env.legal_actions()
            legal_mask = torch.zeros(self.agent.action_dim, dtype=torch.bool, device=self.device)
            for a in legal:
                legal_mask[a] = True

            if turn % 2 == 0:
                if random.random() < explore_eps:
                    action = random.choice(legal)
                else:
                    action, hx_agent = self.agent.get_action(obs, hx_agent, legal_mask=legal_mask, deterministic=False)
                new_obs, _, done, _ = self.env.step(action, player=self.env.AGENTE)
                buffer.obs.append(obs)
                buffer.actions.append(action)
                buffer.rewards.append(0.0)
                buffer.legal.append(legal)
                obs = new_obs
                # ── Shaped reward agresivo: evaluar cada movimiento con Minimax depth 6 ──
                if use_mentor and mentor is not None and not done:
                    clone_env = self.env.clone()
                    val = mentor.evaluate(clone_env)
                    if val <= 0.0:
                        r_shaped = 0.5   # jugada buena (mentor en desventaja o tablas)
                    elif val > 0.1:
                        r_shaped = -0.3  # jugada mala (mentor con ventaja clara)
                    else:
                        r_shaped = 0.0   # neutro (ventaja marginal)
                    buffer.rewards[-1] += r_shaped
            else:
                if use_mentor and mentor is not None:
                    action = mentor.get_action(self.env, legal, pity=pity, depth=mentor_depth)
                elif self.shadow is not None:
                    action, hx_opp = self.shadow.get_action(obs, hx_opp, legal_mask=legal_mask)
                else:
                    action = random.choice(legal)
                _, _, done, _ = self.env.step(action, player=self.env.HUMANO)
            turn += 1

            if done:
                raw = self.env.resultado() if hasattr(self.env, "resultado") else 0.0
                if raw == 1:
                    final_reward = 1.0   # victoria
                elif raw == -1:
                    final_reward = -0.5  # derrota (menos punitiva que -1)
                else:
                    final_reward = 0.3   # empate (bonus positivo)
                if buffer.rewards:
                    buffer.rewards[-1] += final_reward

        return buffer


# ═══════════════════════════════════════════════════════════════════════════════
#  ENTRENADOR PRINCIPAL
# ═══════════════════════════════════════════════════════════════════════════════

class Trainer:
    """
    Bucle principal de entrenamiento con currículo, auto-juego y
    checkpoint condicional.
    """

    def __init__(self, config: TrainingConfig, env: Any):
        self.cfg = config
        self.env = env
        self.device = torch.device(config.device)

        random.seed(config.seed)
        np.random.seed(config.seed)
        torch.manual_seed(config.seed)

        self.agent = LiquidAgent(
            obs_dim=config.obs_dim,
            action_dim=config.action_dim,
            hidden_dim=config.hidden_dim,
            vocab_size=config.vocab_size,
            lora_rank=config.lora_rank,
        ).to(self.device)

        # ── Propagar configuración System 1/2 al agente ──
        self.agent.s12_enabled = config.s12_enabled
        self.agent.s12_max_iters = config.s12_max_iters

        self.shadow = ShadowAgent(self.agent, tau=config.tau)
        self.mentor = StochasticMinimax(mentor_player=-1)
        self.curriculum = CurriculumManager(config)
        self.health = HealthChecker(self.agent, config)
        self.optimizer = torch.optim.AdamW(
            self.agent.parameters(),
            lr=config.phase0_lr,
            weight_decay=config.weight_decay,
        )

        self.game = 0
        self.fase_actual = -1
        self.last_lm_loss = 0.0
        # --- Plateau detection for M6 ---
        self.best_wr_m6 = 0.0
        self.evals_without_improvement = 0
        self.stats: dict[str, list[float]] = {
            "wr_shadow": [],
            "draw_rate": [],
            "grad_norm": [],
            "entropy": [],
            "loss": [],
            "phase": [],
            "wr_m6": [],
        }

    def _load_spanish_corpus(self) -> str:
        candidates = [
            "spanish_books.txt",
            "spanish_wiki.txt",
            "spanish_conversations.jsonl",
        ]
        texts = []
        for fname in candidates:
            path = os.path.join(".", fname)
            if os.path.exists(path):
                with open(path, encoding="utf-8", errors="replace") as f:
                    if fname.endswith(".jsonl"):
                        for line in f:
                            try:
                                data = json.loads(line)
                                texts.append(data.get("text", data.get("output", "")))
                            except Exception:
                                pass
                    else:
                        texts.append(f.read())
        return "\n\n".join(texts)

    def _reset_optimizer_lr(self, lr: float):
        for pg in self.optimizer.param_groups:
            pg["lr"] = lr

    def _evaluate_wr(self, n: int = 50, phase: int = 1) -> tuple[float, float]:
        mentor_eval = StochasticMinimax(mentor_player=-1, depth=6)
        wins = 0
        draws = 0
        for _ in range(n):
            hx_a = torch.zeros(1, self.cfg.hidden_dim, device=self.device)
            obs = self.env.reset()
            done = False
            turn = 0
            while not done:
                legal = self.env.legal_actions()
                legal_mask = torch.zeros(self.cfg.action_dim, dtype=torch.bool, device=self.device)
                for a in legal:
                    legal_mask[a] = True
                if turn % 2 == 0:
                    action, hx_a = self.agent.get_action(obs, hx_a, legal_mask=legal_mask, deterministic=True)
                    obs, _, done, _ = self.env.step(action, player=self.env.AGENTE)
                else:
                    action = mentor_eval.get_action(self.env, legal, pity=0.0)
                    obs, _, done, _ = self.env.step(action, player=self.env.HUMANO)
                turn += 1
            result = self.env.resultado()
            if result == 1:
                wins += 1
            elif result == 0:
                draws += 1
        return wins / max(1, n), draws / max(1, n)

    def train(self, num_games: int) -> LiquidAgent:
        sep = "━" * 58
        t0 = time.time()

        # --- Cargar corpus de lenguaje ---
        tokenizer = SpanishTokenizer()
        raw_text = self._load_spanish_corpus()
        text_ids = tokenizer.encode(raw_text) if raw_text else []
        if text_ids:
            print(f"  Corpus español: {len(text_ids):,} tokens  (vocab={tokenizer.vocab_size})")
        else:
            print("  ⚠️  Sin corpus español — lenguaje desactivado")

        print(f"\n{sep}")
        print(f"  {'ENTRENAMIENTO LNN — AUTO-JUEGO CON CURRÍCULO':^56}")
        print(f"{sep}")

        while self.game < num_games:
            self.game += 1
            game = self.game

            # ── Hardening tick: check if hardening period just ended ──
            if self.curriculum.tick_hardening():
                print(f"\n  ✓ Hardening complete, resuming self-play\n")

            phase = self.curriculum.get_phase(game)
            info = self.curriculum.get_phase_info(game)
            lr = self.curriculum.get_lr(game)
            entropy_coef = self.curriculum.get_entropy_coef(game)
            mentor_pity_val = self.curriculum.get_mentor_pity(game, num_games)
            mentor_prob = self.curriculum.get_mentor_prob(game, num_games)
            mentor_depth = self.curriculum.get_mentor_depth(game, num_games)

            self._reset_optimizer_lr(lr)

            # ── Seleccionar oponente (blended vs fases legacy) ──
            use_shadow = info.use_shadow
            use_mentor = info.use_mentor
            opp_name = info.name
            opp_pity = 0.0
            opp_depth = mentor_depth

            # ── Selección de oponente (4-phase híbrido o blended legacy) ──
            if self.cfg.curriculum_blend:
                opp_cfg = self.curriculum.select_opponent(game, num_games)
                use_shadow = opp_cfg["use_shadow"]
                use_mentor = opp_cfg["use_mentor"]
                opp_name = opp_cfg["opp_name"]
                opp_depth = opp_cfg["opp_depth"]
                opp_pity = opp_cfg["opp_pity"]

            if phase != self.fase_actual:
                self.fase_actual = phase
                print(f"\n{sep}")
                print(f"  FASE {info.id}: {info.name}")
                if self.cfg.curriculum_blend:
                    print(f"  LR = {lr:.6f}  |  β_ent = {entropy_coef:.4f}"
                          f"  |  Opp: {info.opponent}"
                          f"  |  Pity = {mentor_pity_val:.2f}")
                else:
                    print(f"  LR = {lr}  |  β_ent = {entropy_coef:.4f}"
                          f"{f'  | Mentor pity = {mentor_pity_val:.2f}' if phase == 2 else ''}")
                print(f"{sep}\n")

            # --- Jugar partida ---
            sp = SelfPlayModule(
                agent=self.agent,
                shadow=self.shadow if use_shadow else None,
                mentor=self.mentor if use_mentor else None,
                env=self.env,
                device=self.device,
                phase=phase,
            )

            buffer = sp.play_episode(
                use_mentor=use_mentor,
                pity=opp_pity,
                mentor=self.mentor if use_mentor else None,
                mentor_depth=opp_depth,
                explore_eps=self.cfg.phase2_explore_eps if not self.cfg.curriculum_blend else 0.0,
            )

            # --- Actualizar red ---
            if buffer.length > 0:
                loss_dict = compute_reinforce_loss(
                    self.agent, buffer, self.cfg.gamma, entropy_coef, self.device,
                    s12_lambda=self.cfg.s12_lambda,
                    gae_lambda=self.cfg.gae_lambda,
                )
                self.optimizer.zero_grad()
                loss_dict["total"].backward()

                total_norm = 0.0
                for p in self.agent.parameters():
                    if p.grad is not None:
                        total_norm += p.grad.norm(2).item() ** 2
                grad_norm = total_norm ** 0.5

                torch.nn.utils.clip_grad_norm_(
                    self.agent.parameters(), max_norm=self.cfg.clip_grad_norm
                )
                self.optimizer.step()

                # --- Actualizar sombra (EMA continua) ---
                if self.cfg.curriculum_blend and phase >= 1:
                    self.shadow.update(self.agent)  # Shadow se actualiza SIEMPRE en F1+
                elif use_shadow:
                    self.shadow.update(self.agent)

                # --- Entrenamiento de lenguaje (alternado) ---
                if text_ids and self.cfg.text_every_n > 0 and game % self.cfg.text_every_n == 0:
                    L = self.cfg.text_batch_len
                    start = (game * L) % (len(text_ids) - L - 1)
                    chunk = text_ids[start:start + L]
                    tokens = torch.tensor([chunk], dtype=torch.long, device=self.device)
                    self.optimizer.zero_grad()
                    lm_loss_tup = compute_language_loss(self.agent, tokens, self.device, dt=self.cfg.dt_lm, s12_lambda=self.cfg.s12_lambda)
                    lm_loss = lm_loss_tup[0] if isinstance(lm_loss_tup, tuple) else lm_loss_tup
                    lm_loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        self.agent.parameters(), max_norm=self.cfg.clip_grad_norm
                    )
                    self.optimizer.step()
                    self.last_lm_loss = lm_loss.item()

                # --- Evaluación ---
                if game % self.cfg.eval_interval == 0:
                    wr, dr = self._evaluate_wr(n=self.cfg.eval_episodes, phase=phase if not self.cfg.curriculum_blend else 2)
                    entropy = loss_dict["entropy"].item()
                    self.stats["wr_shadow"].append(wr)
                    self.stats["draw_rate"].append(dr)
                    self.stats["grad_norm"].append(grad_norm)
                    self.stats["loss"].append(loss_dict["total"].item())
                    self.stats["entropy"].append(entropy)
                    self.stats["phase"].append(phase)
                    self.stats["wr_m6"].append(wr)

                    # ── Entropy reset: si H < 1.0, inyectar ruido en policy head ──
                    if game % 500 == 0 and entropy < 1.0:
                        with torch.no_grad():
                            for p in self.agent.policy_head.parameters():
                                noise = torch.randn_like(p.data) * 0.05
                                p.data.add_(noise)
                        print(f"  ↻ ENTROPY RESET: H={entropy:.3f} < 1.0 → policy head perturbed")

                    # ── M6 Plateau Detection (only when not already hardening) ──
                    if not self.curriculum.hardening_active:
                        if wr > self.best_wr_m6 + self.cfg.hardening_wr_delta:
                            self.best_wr_m6 = wr
                            self.evals_without_improvement = 0
                        else:
                            self.evals_without_improvement += 1

                        if self.evals_without_improvement >= self.cfg.hardening_patience_evals:
                            self.curriculum.activate_hardening()
                            self.evals_without_improvement = 0
                            print(f"\n  ⚡ HARDENING: {self.cfg.hardening_games} games vs M6 (pity={self.cfg.hardening_pity})\n")

                    lm_tag = f"  LM={self.last_lm_loss:.2f}" if text_ids and self.cfg.text_every_n > 0 else ""
                    htag = " [H]" if self.curriculum.hardening_active else ""
                    print(
                        f"  [{game:5d}/{num_games}] F{phase}: {opp_name:6s}{htag} "
                        f"(d={opp_depth})  "
                        f"WR={wr*100:5.1f}% DR={dr*100:5.1f}%  "
                        f"BestM6={self.best_wr_m6*100:.1f}%  "
                        f"∥g∥={grad_norm:6.2f}  H≈{entropy:.3f}  "
                        f"loss={loss_dict['total'].item():.3f}{lm_tag}  "
                        f"Pity={opp_pity:.2f}  ⏱ {time.time()-t0:.0f}s"
                    )

                    # --- Checkpoint condicional ---
                    healthy = self.health.save_if_healthy(wr, grad_norm, entropy)
                    if healthy:
                        print(f"  ✓ Checkpoint saludable guardado (WR={wr*100:.1f}%)")

        # --- Reporte final detallado ---
        elapsed = time.time() - t0
        mins, secs = divmod(int(elapsed), 60)
        horas = mins // 60; mins = mins % 60
        print(f"\n{sep}")
        print(f"  {'REPORTE FINAL — ENTRENAMIENTO LNN TTT':^56}")
        print(f"{sep}")
        print(f"  Partidas jugadas:        {self.game:>8}")
        print(f"  Tiempo total:            {horas}h {mins}m {secs}s")
        print(f"  Fases completadas:       Fase 0→1→2" if self.fase_actual == 2 else
              f"  Fase alcanzada:          Fase {self.fase_actual}")
        if self.stats["wr_shadow"]:
            wr_final = self.stats["wr_shadow"][-1] * 100
            wr_max = max(self.stats["wr_shadow"]) * 100
            wr_min_val = min(self.stats["wr_shadow"]) * 100
            oponente_eval = self.stats.get("phase", [0])[-1]
            opo_nombre = "MINIMAX" if oponente_eval == 2 else "SHADOW"
            print(f"\n  {'RENDIMIENTO VS ' + opo_nombre:^56}")
            print(f"  Win Rate final:          {wr_final:>7.1f}%")
            print(f"  Win Rate máximo:         {wr_max:>7.1f}%")
            print(f"  Win Rate mínimo:         {wr_min_val:>7.1f}%")
            if self.stats["draw_rate"]:
                dr_final = self.stats["draw_rate"][-1] * 100
                dr_max = max(self.stats["draw_rate"]) * 100
                print(f"  Draw Rate final:         {dr_final:>7.1f}%")
                print(f"  Draw Rate máximo:        {dr_max:>7.1f}%")
            if self.stats["loss"]:
                loss_final = self.stats["loss"][-1]
                loss_min = min(self.stats["loss"])
                print(f"\n  {'PÉRDIDA (LOSS)':^56}")
                print(f"  Loss final:              {loss_final:>8.4f}")
                print(f"  Loss mínimo:             {loss_min:>8.4f}")
            if self.stats["grad_norm"]:
                grad_prom = sum(self.stats["grad_norm"]) / len(self.stats["grad_norm"])
                grad_max = max(self.stats["grad_norm"])
                print(f"  ∥g∥ promedio:             {grad_prom:>8.2f}")
                print(f"  ∥g∥ máximo:               {grad_max:>8.2f}")
            if self.stats["entropy"]:
                ent_prom = sum(self.stats["entropy"]) / len(self.stats["entropy"])
                ent_final = self.stats["entropy"][-1]
                print(f"  Entropía promedio:       {ent_prom:>8.3f}")
                print(f"  Entropía final:          {ent_final:>8.3f}")
            print(f"\n  {'CHECKPOINTS':^56}")
            if self.health.has_checkpoint():
                print(f"  Checkpoint saludable:    {self.health.get_checkpoint_path()}")
            if self.health.best_wr > 0:
                print(f"  Mejor checkpoint:        {self.health.get_best_path()}")
                print(f"  Mejor WR registrado:     {self.health.best_wr*100:.1f}%")
            else:
                print(f"  No se guardaron checkpoints (no se cumplieron condiciones)")
            print(f"\n  {'DIAGNÓSTICO':^56}")
            if self.stats["wr_shadow"]:
                u = wr_final
                if oponente_eval == 2:
                    # Fase 2: evaluado contra Minimax invencible
                    if u > 5:
                        print(f"  ✅ WR > 5% contra Minimax — el agente gana algunas partidas")
                    elif self.stats["draw_rate"] and self.stats["draw_rate"][-1] * 100 > 10:
                        print(f"  ✅ Draw Rate alto ({self.stats['draw_rate'][-1]*100:.1f}%) — el agente aprende a defenderse")
                    else:
                        print(f"  ❌ WR y DR muy bajos contra Minimax — reward shaping insuficiente")
                else:
                    if u > 80:
                        print(f"  ✅ WR muy alto — la IA domina, posible sobreajuste a shadow")
                    elif u > 55:
                        print(f"  ✅ WR competitivo — buen balance exploración/explotación")
                    elif u > 40:
                        print(f"  ⚠️  WR apenas sobre azar — necesita más entrenamiento")
                    else:
                        print(f"  ❌ WR bajo — revisar hiperparámetros o reward")
                if self.stats["entropy"] and self.stats["entropy"][-1] < 0.05:
                    print(f"  ⚠️  Entropía muy baja — política casi determinista")
                elif self.stats["entropy"] and self.stats["entropy"][-1] > 1.0:
                    print(f"  ⚠️  Entropía alta — política aún muy aleatoria")
                if self.stats["grad_norm"] and self.stats["grad_norm"][-1] > 10:
                    print(f"  ⚠️  Gradiente alto — posible inestabilidad")
        else:
            print(f"  No se completaron evaluaciones.")
        print(f"{sep}\n")

        # Guardar copia en ttt_model.pth para compatibilidad con play_ttt.py
        ttt_path = "ttt_model.pth"
        torch.save(self.agent.state_dict(), ttt_path)
        print(f"  Modelo guardado en:      {ttt_path}  (python3 play_ttt.py)")

        return self.agent

    # ─────────────────────────────────────────────────────────────────────────
    #  FASE 2: ENTRENAMIENTO DE LENGUAJE CON REPLAY TTT
    # ─────────────────────────────────────────────────────────────────────────

    def _play_ttt_replay_game(self, mentor_pity: float = 0.0) -> EpisodeBuffer:
        """Juega una partida TTT contra StochasticMinimax depth=6 para replay."""
        sp = SelfPlayModule(
            agent=self.agent, shadow=None, mentor=self.mentor,
            env=self.env, device=self.device, phase=2,
        )
        return sp.play_episode(
            use_mentor=True, pity=mentor_pity, mentor=self.mentor,
            explore_eps=self.cfg.phase2_explore_eps,
        )

    def _phase2_update_reinforce(self, replay: ReplayBuffer) -> float:
        """Consolida REINFORCE loss sobre todo el buffer de replay.

        Si el backbone está completamente congelado (LoRA mode), la pérdida
        no tiene grad_fn y se salta la retropropagación — los TTT games se
        juegan igual para mantener el buffer y la telemetría.
        """
        if len(replay) == 0:
            return 0.0
        total_loss = 0.0
        count = 0
        can_train = False
        for ep in replay.sample():
            if ep.length == 0:
                continue
            loss_dict = compute_reinforce_loss(
                self.agent, ep, self.cfg.gamma,
                entropy_coef=0.05,
                device=self.device,
                s12_lambda=self.cfg.s12_lambda,
                gae_lambda=self.cfg.gae_lambda,
            )
            total_loss += loss_dict["total"].item()
            if loss_dict["total"].grad_fn is not None:
                if not can_train:
                    self.optimizer.zero_grad()
                    can_train = True
                loss_dict["total"].backward()
            count += 1
        if can_train:
            torch.nn.utils.clip_grad_norm_(
                [p for p in self.agent.parameters() if p.requires_grad],
                max_norm=self.cfg.clip_grad_norm,
            )
            self.optimizer.step()
        return total_loss / count

    def train_phase2(self, checkpoint_path: str, lm_epochs: int):
        """
        Fase 2: carga un checkpoint TTT pre-entrenado, congela policy/value,
        y entrena lenguaje con replay periódico de TTT.

        Args:
            checkpoint_path: ruta al .pth con pesos TTT pre-entrenados
            lm_epochs: número de épocas sobre el corpus de lenguaje
        """
        sep = "━" * 58
        t0 = time.time()

        # ── Cargar checkpoint ──
        print(f"\n{sep}")
        print(f"  {'FASE 2: LENGUAJE + REPLAY TTT':^56}")
        print(f"{sep}")
        print(f"  Cargando checkpoint: {checkpoint_path}")
        sd = torch.load(checkpoint_path, map_location=self.device)
        self.agent.load_state_dict(sd, strict=False)
        print(f"  ✓ Checkpoint cargado ({len(sd)} keys, strict=False)")

        # ── Congelar cabezas de TTT (nunca se descongelan) ──
        frozen = 0
        for name, param in self.agent.named_parameters():
            if 'policy_head' in name or 'value_head' in name:
                param.requires_grad = False
                frozen += 1
        print(f"  ✓ Cabezas congeladas: {frozen} parámetros (policy + value) — nunca se tocan")

        # ── Si LoRA está activo: congela TODO el backbone, solo entrena LoRA + lenguaje ──
        lora_mode = self.agent.lora_rank > 0
        if lora_mode:
            # Congelar TODO primero
            for param in self.agent.parameters():
                param.requires_grad = False
            # Descongelar solo LoRA + lenguaje
            for name, param in self.agent.named_parameters():
                if 'lora_' in name or 'lang_head' in name or 'lang_norm' in name or 'text_embed' in name:
                    param.requires_grad = True
            print(f"  ✓ LoRA rank={self.agent.lora_rank}: backbone COMPLETAMENTE congelado — solo entrena LoRA + lenguaje")

        trainable = [p for p in self.agent.parameters() if p.requires_grad]
        n_trainable = sum(p.numel() for p in trainable)
        n_total = sum(p.numel() for p in self.agent.parameters())
        print(f"  Parámetros entrenables: {n_trainable:,} / {n_total:,} totales")
        print(f"  dt_lm={self.cfg.dt_lm}  dt_ttt={self.cfg.dt_ttt}  α={self.cfg.lm_alpha}")
        print(f"  replay_interval={self.cfg.replay_interval}  buffer_size={self.cfg.replay_buffer_size}")
        print(f"{sep}\n")

        # ── Optimizer solo para parámetros entrenables ──
        self.optimizer = torch.optim.AdamW(
            trainable,
            lr=self.cfg.text_lm_lr,
            weight_decay=self.cfg.weight_decay,
        )

        # ── Cargar corpus ──
        tokenizer = SpanishTokenizer()
        raw_text = self._load_spanish_corpus()
        text_ids = tokenizer.encode(raw_text) if raw_text else []
        if not text_ids:
            print("  ❌ Sin corpus español — abortando Fase 2")
            return
        print(f"  Corpus: {len(text_ids):,} tokens  (vocab={tokenizer.vocab_size})")

        # ── Buffer de replay ──
        replay = ReplayBuffer(maxlen=self.cfg.replay_buffer_size)
        mentor_pity = 0.05  # pity fijo bajo en Fase 2 (solo para mantener variedad)

        # ── Métricas ──
        L = self.cfg.text_batch_len
        batches_per_epoch = 2000  # batches aleatorios por época (no barre todo el corpus)
        total_batches = lm_epochs * batches_per_epoch
        step = 0
        best_lm_loss = float('inf')
        lm_losses: list[float] = []
        reinforce_losses: list[float] = []
        rl_loss: float = 0.0
        wr: float = 0.0
        dr: float = 0.0

        print(f"  Batch budget: {total_batches:,} batches "
              f"({batches_per_epoch} por época × {lm_epochs} épocas)")
        s12_tag = 'SÍ' if self.cfg.s12_enabled else 'NO'
        print(f"  S12 para LM:   {s12_tag} (α={self.cfg.lm_alpha})")
        print(f"  Policy/Value congelados — TTT no se degrada")
        print(f"{sep}\n")

        # ── Diagnóstico de parámetros (backbone congelado) ──
        n_train = sum(p.numel() for p in self.agent.parameters() if p.requires_grad)
        n_total_p = sum(p.numel() for p in self.agent.parameters())
        pct = 100.0 * n_train / max(n_total_p, 1)
        print(f"  📊 Diagnóstico de parámetros:")
        print(f"     Entrenables: {n_train:,} / {n_total_p:,} ({pct:.2f}%)")
        frozen_names = [n for n, p in self.agent.named_parameters() if not p.requires_grad]
        if frozen_names:
            print(f"     Congelados ({len(frozen_names)} capas): {frozen_names[0]} ... {frozen_names[-1]}")
        print()

        # ── Inicializar telemetría ──
        telemetry = TrainingTelemetry()
        eval_interval = max(1, batches_per_epoch // 4)  # 4 evaluaciones por época

        # ── Bucle principal: batches aleatorios ──
        for step in range(1, total_batches + 1):
            epoch = (step - 1) // batches_per_epoch + 1

            # ─── PASO DE LENGUAJE (fragmento secuencial del corpus) ───
            start = ((step - 1) * (L // 2)) % (len(text_ids) - L)
            chunk = text_ids[start:start + L]
            tokens = torch.tensor([chunk], dtype=torch.long, device=self.device)

            # S12 desactivado durante LM forward (mucho más rápido)
            old_s12 = self.agent.s12_enabled
            self.agent.s12_enabled = False
            lm_loss, lm_effort = compute_language_loss(
                self.agent, tokens, self.device, dt=self.cfg.dt_lm,
                s12_lambda=0.0,
            )
            self.agent.s12_enabled = old_s12
            total_loss = self.cfg.lm_alpha * lm_loss

            self.optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                trainable, max_norm=self.cfg.clip_grad_norm,
            )
            self.optimizer.step()

            # ─── REPLAY TTT (cada replay_interval batches) ───
            if step % self.cfg.replay_interval == 0:
                # Re-activar S12 para TTT si está configurado
                old_s12 = self.agent.s12_enabled
                self.agent.s12_enabled = self.cfg.s12_enabled
                buffer = self._play_ttt_replay_game(mentor_pity=mentor_pity)
                replay.add(buffer)
                rl_loss = self._phase2_update_reinforce(replay)
                self.agent.s12_enabled = old_s12

            # ─── LOG cada 100 steps ───
            if step % 100 == 0:
                elapsed = time.time() - t0
                avg_lm = lm_loss.item()
                lm_tag = f"  RL_loss={rl_loss:.4f}" if step % self.cfg.replay_interval == 0 else ""
                print(
                    f"  [E{epoch:3d}/{lm_epochs} step{step:6d}/{total_batches}] "
                    f"LM={avg_lm:.4f}  "
                    f"replay={len(replay)}{lm_tag}  ⏱ {elapsed:.0f}s"
                )

            # ─── TELEMETRÍA: registro periódico ───
            if step % eval_interval == 0:
                lm_val = lm_loss.item()
                wr_val, dr_val = self._evaluate_wr(n=20, phase=2)
                # effort/iters solo si S12 activo (TTT replay steps)
                effort_val = 0.0
                iters_val = 0.0
                telemetry.log(step, loss_lang=lm_val, wr=wr_val, dr=dr_val,
                              effort=effort_val, iters=iters_val)
                # Log cada 4 evaluaciones (no saturar)
                if step % (eval_interval * 4) == 0:
                    print(f"     📈 tele: LM={lm_val:.4f}  WR={wr_val*100:.1f}%  DR={dr_val*100:.1f}%")

            # ─── EVALUACIÓN fin de época ───
            if step % batches_per_epoch == 0:
                lm_losses.append(avg_lm)

                wr, dr = self._evaluate_wr(n=50, phase=2)
                elapsed = time.time() - t0
                print(
                    f"\n  ─── EPOCH {epoch}/{lm_epochs} ({batches_per_epoch} batches) ───\n"
                    f"  LM loss = {avg_lm:.4f}  "
                    f"WR vs M6 = {wr*100:.1f}%  DR = {dr*100:.1f}%  "
                    f"⏱ {elapsed:.0f}s\n"
                )

                if avg_lm < best_lm_loss:
                    best_lm_loss = avg_lm
                    ckpt_path = os.path.join(self.cfg.checkpoint_dir, "phase2_best.pth")
                    torch.save(self.agent.state_dict(), ckpt_path)
                    print(f"  ✓ Nuevo mejor checkpoint LM: {ckpt_path} (loss={avg_lm:.4f})\n")

        # ─── Exportar telemetría ───
        telemetry.save("telemetria_fase2.json")

        # ─── REPORTE FINAL FASE 2 ───
        elapsed = time.time() - t0
        mins, secs = divmod(int(elapsed), 60)
        horas = mins // 60
        mins = mins % 60
        print(f"\n{sep}")
        print(f"  {'REPORTE FINAL — FASE 2 (LENGUAJE + REPLAY)':^56}")
        print(f"{sep}")
        print(f"  Batch budget total:      {total_batches:,}")
        print(f"  Épocas completadas:      {lm_epochs}")
        print(f"  Tiempo total:            {horas}h {mins}m {secs}s")
        if lm_losses:
            print(f"  LM loss final:           {lm_losses[-1]:.6f}")
            print(f"  LM loss mínimo:          {min(lm_losses):.6f}")
        if wr is not None:
            print(f"  Win Rate vs M6:          {wr*100:.1f}%")
            print(f"  Draw Rate vs M6:         {dr*100:.1f}%")
        print(f"{sep}\n")

        # Guardar modelo final
        final_path = "ttt_model_phase2.pth"
        torch.save(self.agent.state_dict(), final_path)
        print(f"  Modelo Fase 2 guardado: {final_path}")

        return self.agent


# ═══════════════════════════════════════════════════════════════════════════════
#  MODO OVERNIGHT (7h TTT + 1h grid)
# ═══════════════════════════════════════════════════════════════════════════════

def entrenar_overnight(config: TrainingConfig) -> None:
    sep = "━" * 58
    TTT_HORAS = 7 * 3600
    GRID_HORAS = 1 * 3600
    t0 = time.time()

    ruta_ttt = os.path.join(config.checkpoint_dir, "lnn_ttt_overnight.pth")
    os.makedirs(config.checkpoint_dir, exist_ok=True)

    print(f"\n{sep}")
    print(f"  {'[OVERNIGHT] BLOQUE 1: TIC-TAC-TOE (7 HORAS)':^56}")
    print(f"{sep}\n")

    env = TicTacToeEnv()
    trainer = Trainer(config, env)
    trainer.train(50000)

    torch.save(trainer.agent.state_dict(), ruta_ttt)
    print(f"  [OVERNIGHT] Checkpoint guardado en {ruta_ttt}")

    print(f"\n{sep}")
    print(f"  {'[OVERNIGHT] BLOQUE 2: EVALUACIÓN DE GRILLA (1 HORA)':^56}")
    print(f"{sep}\n")

    os.environ.pop("SDL_VIDEODRIVER", None)
    import subprocess

    grid_proc = subprocess.Popen(
        [sys.executable, "-u", "main_agent_loop.py"],
        cwd=os.path.dirname(os.path.abspath(__file__)),
    )
    try:
        grid_proc.wait(timeout=GRID_HORAS)
    except subprocess.TimeoutExpired:
        grid_proc.terminate()
        grid_proc.wait()

    elapsed = time.time() - t0
    print(f"\n{sep}")
    print(f"  {'ENTRENAMIENTO NOCTURNO COMPLETO':^56}")
    print(f"{sep}")
    print(f"  Tiempo total:  {elapsed/3600:.1f}h ({elapsed:.0f}s)")
    print(f"  Checkpoint:    {ruta_ttt}")
    print(f"{sep}\n")


# ═══════════════════════════════════════════════════════════════════════════════
#  PUNTO DE ENTRADA
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Entrenamiento LNN con auto-juego y currículo")
    parser.add_argument("--overnight", action="store_true",
                        help="Modo nocturno: 7h TTT + 1h grid")
    parser.add_argument("--games", type=int, default=10000,
                        help="Número de partidas (modo normal)")
    parser.add_argument("--device", type=str, default="cpu",
                        help="Dispositivo (cpu, cuda, mps)")
    parser.add_argument("--no-gui", action="store_true",
                        help="No abrir interfaz gráfica al terminar")
    parser.add_argument("--seed", type=int, default=42,
                        help="Semilla aleatoria")
    parser.add_argument("--multitask", action="store_true",
                        help="Entrenar TTT + Lenguaje español simultáneamente")
    # --- Fase 2: LM post-TTT con replay ---
    parser.add_argument("--phase2", action="store_true",
                        help="Fase 2: cargar checkpoint TTT y entrenar lenguaje + replay")
    parser.add_argument("--checkpoint", type=str, default="ttt_model.pth",
                        help="Checkpoint TTT pre-entrenado para Fase 2")
    parser.add_argument("--lm-epochs", type=int, default=10,
                        help="Épocas de lenguaje en Fase 2 (cada época = 2000 batches aleatorios)")
    # --- LoRA (Low-Rank Adaptation) para lenguaje ---
    parser.add_argument("--lora-rank", type=int, default=0,
                        help="Rango LoRA para lenguaje (0=desactivado, 8-16 típico)")
    # --- System 1/2 Adaptive Computation ---
    parser.add_argument("--s12", action="store_true",
                        help="Activar System 1/2 (cómputo adaptativo ACT-style)")
    args = parser.parse_args()

    config = TrainingConfig(seed=args.seed, device=args.device)
    if args.lora_rank > 0:
        config.lora_rank = args.lora_rank
    if args.s12:
        config.s12_enabled = True
    if not args.multitask:
        # desactivar lenguaje si no se pide --multitask
        config.text_every_n = 0

    if args.overnight:
        entrenar_overnight(config)
        return

    if args.phase2:
        if not os.path.exists(args.checkpoint):
            print(f"  ❌ Checkpoint no encontrado: {args.checkpoint}")
            sys.exit(1)
        env = TicTacToeEnv()
        trainer = Trainer(config, env)
        trainer.train_phase2(args.checkpoint, args.lm_epochs)
        return

    env = TicTacToeEnv()
    trainer = Trainer(config, env)
    model = trainer.train(args.games)

    if not args.no_gui:
        import subprocess
        ckpt = "ttt_model.pth"
        if not os.path.exists(ckpt):
            print("  Error: no se encontró ttt_model.pth")
        else:
            print("  Abriendo GUI TTT vs IA...\n")
            subprocess.run([sys.executable, "gui_ttt.py", "--checkpoint", ckpt])


if __name__ == "__main__":
    main()
