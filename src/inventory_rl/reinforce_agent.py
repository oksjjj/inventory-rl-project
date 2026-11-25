"""REINFORCE + 가치 기준선으로 발주량 전체를 학습한다. (수업 8강)

표 Q[상태][행동]은 쓰지 않는다. 정책 πθ(a|s)를 가우시안 분포로 모델링한다.
세 공급 링크의 연속 발주량을 각각 독립적인 정규분포에서 샘플링한다.
평균 μ_i(s)는 관측에 따라 신경망이 결정하고, 표준편차 σ_i는 링크별로 직접 학습한다.
샘플링한 발주량은 환경에 전달하기 전에 0~80으로 제한한다.

8강 슬라이드 17 (순수 REINFORCE):
    θ ← θ + α ∇ log π(a_t | s_t) × G_t

여기에 슬라이드 19 기준선을 더했다. V(s) 는 정책이 아니라 분산을 줄이는 값이다.
    A_t = G_t − V(s_t)
    θ ← θ + α ∇ log π(a_t | s_t) × A_t

코드에서 손실에 마이너스를 붙이는 이유는 PyTorch 가 경사하강만 해서다.
−log π × A 를 줄이면 log π × A 를 올리는 것과 같다.

네트워크 이름은 ReinforceNet. 매 기간 δ_t 가 아니라 에피소드 G_t 를 쓴다.

한 줄 요약:
    1) 관측으로 발주량 평균 μ와 가치 V를 계산하고, 학습 파라미터로 표준편차 σ를 계산한다.
    2) 발주 ~ Normal(μ, σ) 에서 뽑고 환경을 한 기간 진행한다.
    3) 에피소드(30기간)가 끝나면
           G_t = r_t + γ r_{t+1} + γ² r_{t+2} + ...
           A_t = G_t − V(s_t)
       로 “생각보다 잘했으면” 그 발주의 로그확률을 올린다.
"""

# 이 파일을 읽는 순서
# 1. ReinforceNet.__init__ / forward : 정책 분포와 가치 함수가 어떻게 만들어지는가?
# 2. ReinforceAgent.act_and_log    : 현재 상태에서 발주량을 어떻게 선택하는가?
# 3. ReinforceAgent.update_episode : 30기간의 경험으로 어떻게 학습하는가?
# 4. ReinforceAgent.get_action     : 학습된 정책을 평가할 때는 어떻게 발주하는가?
#
# 전체 학습 루프는 scripts/train_reinforce.py에 있다.
# env.reset → [act_and_log → env.step → 경험 저장] × 30 → update_episode 1회
# DQN과 달리 125개 행동의 Q값을 계산하거나 Replay Buffer에서 경험을 뽑지 않는다.
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.distributions import Normal

from inventory_rl.env import MAX_ORDER, clip_order
from inventory_rl.obs import scale_obs

# 학습을 시작하자마자 평균 발주가 약 20이 되게 한다.
# sigmoid(x)=20/80=0.25 를 만족하는 x 가 logit(0.25)=ln(0.25/0.75).
# 역변환을 구하는 이유: 신경망의 출력 z에 sigmoid를 적용하고 80을 곱한다.
# μ = 80 × sigmoid(z), 초기 μ=20을 원하므로 sigmoid(z)=0.25여야 한다.
# z=log(p/(1-p))=log(1/3)≈-1.099를 초기 bias로 쓰면 원하는 평균이 나온다.
# 평균 발주량 20은 학습 시작점일 뿐, 이후에는 상태와 학습 결과에 따라 달라진다.
_INIT_ORDER_LOGIT = math.log(0.25 / 0.75)


class ReinforceNet(nn.Module):
    """REINFORCE 정책망: 가우시안 정책 πθ(a|s)와 가치 기준선 V(s)를 구성한다.

    공통 층(backbone)을 통과한 뒤
      mean_head  → 관측에 따른 세 링크의 가우시안 평균 μ_i(s)
      value_head → “이 관측에서 앞으로 받을 보상” 예측 V(s)
    로 갈라진다. 표준편차 σ_i는 별도의 학습 파라미터 log_std로 결정한다.
    정책은 세 독립 정규분포의 결합분포 πθ(a|s) = ∏_i N(a_i; μ_i(s), σ_i²)이다.
    V(s)는 가우시안 정책의 구성 요소가 아니라 G_t의 기준선이다.
    매 기간 r + γV(s') 로 바꾸지 않는다. 그 구현은 ActorCriticAgent 이다.
    """

    def __init__(self, obs_dim: int, act_dim: int, hidden: int = 64, max_order: float = MAX_ORDER) -> None:
        super().__init__()
        self.max_order = max_order
        # 관측 15칸 → 64 → 64. Tanh 는 -1~1 로 눌러 준다.
        # backbone은 수요, 재고, 입고 예정 물량 등 15개 관측값을 받아
        # 정책과 가치 함수가 공통으로 활용할 64개의 특성(feature)을 만든다.
        # Linear(in, out)는 y = Wx+b, Tanh는 비선형 활성화 함수다.
        # 비선형 함수가 없으면 여러 Linear 층을 쌓아도 하나의 선형 변환과 같다.
        # 배치 크기를 B라고 하면 (B,15) → (B,64) → (B,64).
        # 여기서 64는 관측이나 행동의 수가 아니라 은닉층의 뉴런 수다.
        self.backbone = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
        )
        # [가우시안 정책의 평균 μ(s)]
        # 관측 s를 입력받아 세 공급 링크의 평균 발주량을 결정한다.
        # forward()에서 sigmoid × MAX_ORDER를 적용하여 평균을 0~80으로 제한한다.
        self.mean_head = nn.Linear(hidden, act_dim)  # 출력 3칸 = 가우시안 평균 3개
        # [가치 기준선 V(s)] 정책 분포와는 별도로 미래 할인 누적 보상을 예측한다.
        self.value_head = nn.Linear(hidden, 1)  # 출력 1칸 = V(s)
        # [가우시안 정책의 표준편차 σ]
        # 각 링크의 발주량 a_i ~ Normal(μ_i(s), σ_i)에서 독립적으로 샘플링한다.
        # PyTorch Normal의 두 번째 인자는 분산 σ²가 아니라 표준편차 σ이다.
        # σ는 양수여야 하므로 제약 없는 log_std를 학습하고 σ = exp(log_std)로 변환한다.
        # nn.Parameter: 관측을 입력받지 않는 직접 학습 변수. 상태와 무관하지만 학습 중 갱신된다.
        # 세 링크의 log_std는 [1, 1, 1]로 초기화되지만 각 원소는 독립적으로 학습된다.
        # 초기 σ_i = exp(1) ≈ 2.72. 평균이 20이면 약 68%가 17.28~22.72 사이에서 샘플링된다.
        # forward()에서 σ의 범위를 0.05~12로 제한하여 탐색 크기를 조절한다.
        self.log_std = nn.Parameter(torch.full((act_dim,), 1.0))
        # 가중치는 0, 편향만 logit. 시작 직후 평균 발주 ≈ (20, 20, 20).
        # mean_head의 계산식 z=W×hidden+b에서 W=0, b=logit(0.25)로 설정한다.
        # 처음에는 hidden이 어떤 값이든 z가 일정하므로 세 링크 평균이 모두 20이다.
        # 학습 중 W와 b가 갱신되면 상태마다 서로 다른 평균을 출력할 수 있다.
        # value_head와 backbone은 이 방식으로 초기화하지 않는다(기본 초기화 사용).
        nn.init.zeros_(self.mean_head.weight)
        nn.init.constant_(self.mean_head.bias, _INIT_ORDER_LOGIT)

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # forward는 신경망의 순전파다. 여기서는 행동을 아직 샘플링하지 않는다.
        # obs=(B,15), hidden=(B,64), mean=(B,3), std=(3,), value=(B,)가 된다.
        # B=1은 현재 한 시점의 관측 하나를 처리한다는 뜻이다.
        hidden = self.backbone(obs)
        # 가우시안 정책의 상태별 평균 μ(s): sigmoid × MAX_ORDER로 0~80 범위에 맞춘다.
        # 예: sigmoid=0.25 → mean=80*0.25=20
        # 예: mean_head 출력이 [-1.099, 0, 1.099]라면
        # sigmoid는 약 [0.25, 0.5, 0.75], 평균 발주량은 [20,40,60]이다.
        # 제한되는 것은 분포의 평균이지, 정규분포에서 뽑은 행동 자체는 아니다.
        mean = self.max_order * torch.sigmoid(self.mean_head(hidden))
        # 가우시안 정책의 링크별 표준편차 σ: 작으면 평균 주변을 탐색하고, 크면 넓게 탐색한다.
        # exp()는 음수 log_std도 양수 표준편차로 변환한다.
        # clamp(min,max)는 범위를 벗어난 값을 경계값으로 제한한다.
        # 초기 σ≈2.72는 평균 주변의 좁은 탐험을 의미한다.
        # σ는 상태의 함수가 아니므로 같은 링크에서는 모든 상태에 동일한 σ를 사용한다.
        # 평균과 표준편차 모두 정책 손실의 역전파로 학습된다(σ에는 entropy 손실도 작용).
        # clamp 바깥쪽에서는 해당 표준편차 경로의 기울기가 0이 될 수 있다.
        std = self.log_std.exp().clamp(0.05, 12.0)
        # V(s)는 지금부터 에피소드가 끝날 때까지 얻을 할인 누적 보상의 예측값이다.
        # 세 링크별 가치가 아니라 전체 공급망에 대한 스칼라 값 하나를 출력한다.
        # squeeze(-1)은 마지막 크기 1인 차원을 제거한다. (B,1) → (B,).
        # mean과 std는 정책 π를 구성하고, value는 학습할 때 기준선으로만 활용한다.
        value = self.value_head(hidden).squeeze(-1)  # (batch, 1) → (batch,)
        return mean, std, value


class ReinforceAgent:
    """한 에피소드를 통째로 모은 뒤, 그 결과로 정책을 한 번 바꾼다."""

    def __init__(
        self,
        obs_dim: int = 15,
        act_dim: int = 3,
        lr: float = 3e-4,
        gamma: float = 0.99,
        entropy_coef: float = 0.01,
        reward_scale: float = 50.0,
        device: str | None = None,
    ) -> None:
        # 학습 설정: obs_dim=15, act_dim=3, lr=0.0003, γ=0.99.
        # γ는 미래 보상의 할인율, lr은 Adam이 사용하는 학습률이다.
        # entropy_coef=0.01은 탐험 유지의 가중치이지 ε-greedy의 ε가 아니다.
        # reward_scale=50은 학습용 보상의 크기를 조절하기 위한 상수다.
        # Mac의 Apple Silicon에서는 MPS를 사용하고, 사용 불가 시 CPU를 사용한다.
        if device is None:
            device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.device = torch.device(device)
        self.gamma = gamma  # 할인율. 1에 가까울수록 먼 기간 보상도 지금 발주 탓으로 본다.
        self.entropy_coef = entropy_coef  # entropy 가 클수록 발주를 더 다양하게 뽑는다.
        self.reward_scale = reward_scale  # 원본 보상 합이 수백이라, 나눠야 V 학습이 안정적이다.
        # ReinforceNet에는 backbone, mean_head, value_head, log_std가 모두 포함된다.
        # .to(device)는 신경망의 파라미터를 연산 장치로 이동시킨다.
        self.model = ReinforceNet(obs_dim, act_dim).to(self.device)
        # model.parameters()에는 nn.Linear의 W,b와 직접 등록한 log_std도 들어 있다.
        # Adam 하나가 정책과 가치 함수 및 공유 backbone을 함께 업데이트한다.
        # Adam을 만드는 것만으로 학습되는 것은 아니고, 아래 optimizer.step()에서 갱신된다.
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

    def _tensor_obs(self, obs: np.ndarray) -> torch.Tensor:
        """numpy 관측 → 스케일 → 학습 장치(GPU/MPS/CPU) 텐서."""
        # scale_obs는 obs.py에 정의되어 있다. 관측마다 정해진 상수로 나누어
        # 수요, 재고, 기간 등 서로 크기가 다른 입력의 스케일을 맞춘다.
        # 예: 수요 40/80=0.5, 공장 재고 200/200=1.0, 기간 15/30=0.5.
        # min-max 정규화처럼 항상 0~1 범위를 보장하는 것은 아니다.
        # NumPy (15,)를 float32 PyTorch 텐서 (15,)로 변환한다.
        return torch.as_tensor(scale_obs(obs), dtype=torch.float32, device=self.device)

    def get_action(self, obs: np.ndarray, env, deterministic: bool = True) -> np.ndarray:
        """평가용 발주.

        deterministic=True 이면 평균 μ 만 쓴다. 발표 비교 때 이렇게 한다.
        학습 중이 아니라서 기울기를 계산하지 않는다(torch.no_grad).
        """
        # 평가에서는 학습 시의 확률적 탐험을 제거하고 정책의 평균을 기본적으로 사용한다.
        # eval()은 평가 모드 설정이고, no_grad()는 자동 미분 기록을 끄는 별개의 기능이다.
        self.model.eval()
        with torch.no_grad():
            # unsqueeze(0): (15,) → (1,15). 배치 차원을 추가한다.
            # self.model(...)을 호출하면 ReinforceNet.forward()가 실행된다.
            # 반환값 중 V(s)는 실제 발주 결정에는 필요하지 않아 _로 받는다.
            mean, std, _ = self.model(self._tensor_obs(obs).unsqueeze(0))  # 배치 1
            # deterministic=True: mean[0]=(3,)을 그대로 발주량으로 사용한다.
            # False일 때만 정규분포에서 무작위 행동을 샘플링한다.
            # 학습된 상태별 평균을 쓰는 것이므로 고정량 발주 정책과는 다르다.
            sampled = mean[0] if deterministic else Normal(mean, std).sample()[0]
        # CPU로 이동한 뒤 NumPy 배열로 변환해 Gymnasium 인터페이스에 맞춘다.
        # 평균 또는 샘플링 결과를 0~80으로 제한한다.
        return clip_order(sampled.cpu().numpy())  # 예: [19.8, 22.1, 18.4] → 0~80

    def act_and_log(self, obs: np.ndarray) -> tuple[np.ndarray, torch.Tensor, torch.Tensor, torch.Tensor]:
        """학습용 발주.

        평균만 쓰면 항상 같은 발주만 해서 새로운 행동을 못 찾는다.
        그래서 Normal(μ, σ) 에서 한 점을 뽑는다.

        같이 돌려주는 것:
          log_prob : 그 발주가 나올 로그확률. 나중에 “잘했으면” 이 값을 올린다.
          entropy  : 분포가 얼마나 넓은지. 너무 좁아지면 탐색이 멈춘다.
          value    : 지금 관측의 V(s). 에피소드 끝에 G_t 와 비교한다.
        """
        # [1단계] 관측을 정규화한 뒤 배치 차원을 추가한다. (15,) → (1,15).
        obs_t = self._tensor_obs(obs).unsqueeze(0)  # 배치 차원 1칸 추가
        # [2단계] 순전파로 현재 상태의 정책 분포와 가치 기준선을 계산한다.
        # mean=(1,3), std=(3,), value=(1,).
        # std는 배치 차원이 없어도 PyTorch broadcasting으로 (1,3)에 맞춰 사용된다.
        # 이 시점에는 파라미터가 업데이트되지 않으며 자동 미분 그래프만 기록된다.
        mean, std, value = self.model(obs_t)
        # 정책 πθ(a|s)를 세 링크의 독립 가우시안 분포로 구성한다.
        # 예: mean=[[20,30,40]], std=[2.72,2.72,2.72]이면 각 링크의
        # 평균과 표준편차를 가진 세 개의 독립적인 정규분포가 만들어진다.
        # dist 자체가 확률적 정책이다. Q값의 최댓값을 고르는 DQN과 다르다.
        dist = Normal(mean, std)
        # 평균을 그대로 사용하지 않고 정책 분포에서 연속 행동을 샘플링한다.
        # 실제 환경에는 학습 스크립트에서 clip_order를 적용한 발주량을 전달한다.
        # [3단계] 정책에서 행동을 하나 뽑는다. 결과는 (1,3).
        # sample()은 행동에 대한 경사를 추적하지 않는다. REINFORCE는
        # 행동 자체를 미분하는 대신, 고정된 행동의 log_prob를 미분한다.
        # rsample()과 달리 샘플링 경로를 통해 정책을 미분하지 않는 이유다.
        # 환경에는 clip_order(sampled)를 전달하지만, log_prob는 원래 샘플로 계산한다.
        # 즉, 가우시안은 제한 전 행동의 분포이고 실제 발주량은 제한된 변환 결과다.
        sampled = dist.sample()  # 예: 평균 (20,20,20) 근처에서 (18.1, 23.4, 19.0)
        # 독립 분포의 결합 확률밀도는 각 링크 확률밀도의 곱이다.
        # 로그를 취하면 합이 되므로 세 링크의 로그확률밀도를 합산한다.
        # [4단계] 정책 경사 학습을 위해 선택한 행동의 로그확률밀도를 기록한다.
        # 연속 행동은 특정 실수값 하나의 확률이 아니라 확률밀도를 사용한다.
        # dist.log_prob(sampled)=(1,3): 각 발주량의 로그확률밀도.
        # π(a|s)=∏_i π_i(a_i|s) → log π(a|s)=Σ_i log π_i(a_i|s).
        # sum(dim=-1)로 마지막 차원(3개 링크)을 합치면 (1,)이 된다.
        # sample에는 경사가 없지만 log_prob는 mean과 std에 대한 경사를 유지한다.
        # 따라서 나중에 log_prob × Advantage를 통해 정책을 학습할 수 있다.
        log_prob = dist.log_prob(sampled).sum(dim=-1)
        # [5단계] 분포의 엔트로피(불확실성)를 계산한다. (1,3) → (1,).
        # 정규분포 하나의 엔트로피는 0.5 × log(2πeσ²)이며 σ가 커질수록 커진다.
        # 세 독립 분포의 엔트로피를 합산한다.
        # 나중에 손실에서 엔트로피를 보상하여 탐험이 지나치게 빨리 줄어들지 않게 한다.
        entropy = dist.entropy().sum(dim=-1)
        # [6단계] 환경에 넣을 행동은 미분이 필요 없으므로 detach 후 NumPy로 바꾼다.
        # sampled[0]은 배치 차원 제거: (1,3) → (3,).
        # 하지만 log_prob, entropy, value는 Tensor 그대로 반환해야 학습할 수 있다.
        # train_reinforce.py에서 이 네 결과 중 행동은 env.step에 전달하고,
        # 나머지 Tensor 및 환경 보상은 리스트에 저장해 에피소드 끝까지 보관한다.
        action = sampled[0].detach().cpu().numpy()  # 환경에 넣을 numpy 3칸
        # value.squeeze(0): (1,) → 스칼라(0차원). 나중에 stack으로 묶는다.
        # 이 함수는 매 기간 호출되지만 optimizer.step은 호출하지 않는다.
        return action, log_prob, entropy, value.squeeze(0)

    def update_episode(
        self,
        log_probs: list[torch.Tensor],
        values: list[torch.Tensor],
        rewards: list[float],
        entropies: list[torch.Tensor],
    ) -> dict[str, float]:
        """에피소드 전체 보상으로 정책을 한 번 바꾼다. 여기가 REINFORCE 본체다.

        뒤에서부터 G 를 쌓는다. 마지막 기간은 G = r_T,
        그 앞은 G_t = r_t + γ G_{t+1} 이다.

        A_t = G_t − V(s_t)
          A > 0 : 생각보다 이득 → 그 발주가 더 자주 나오게
          A < 0 : 생각보다 손해 → 그 발주가 덜 나오게
        손실은 −logπ(a|s) × A 의 평균. 마이너스를 붙여 “이득인 행동의 확률을 올리는”
        방향으로 경사하강을 쓴다.
        """
        # [1단계] 각 시점의 Return G_t를 계산한다(30기간의 실제 보상 사용).
        # rewards=[r_0,r_1,...,r_29]이고, G_t는 t부터 마지막까지의 할인 누적 보상이다.
        # G_t = Σ_{k=t}^{29} γ^(k-t) × (r_k / reward_scale).
        # REINFORCE는 다음 상태의 V(s')로 미래를 추정하지 않는다.
        # 전체 에피소드를 실제로 경험한 다음 G_t를 계산하는 Monte Carlo 방식이다.
        # 이 구현은 보상을 50으로 나눈 학습용 G를 사용한다.
        # 평가용 누적 이익(로그의 return)은 할인하지 않은 원본 보상 합이다.
        returns = []
        # 마지막 보상 이후에는 더 이상 받을 보상이 없으므로 G_30=0으로 시작한다.
        discounted = 0.0
        # 뒤에서부터 계산하면 G_t = r_t/50 + γG_(t+1)을 한 번씩만 계산하면 된다.
        # 예: 보상 [10,20,30], γ=0.99이면
        # G_2=30/50=0.6
        # G_1=20/50+0.99×0.6=0.994
        # G_0=10/50+0.99×0.994=1.18406.
        # 먼저 G_2부터 구하고 insert(0,...)로 [G_0,G_1,G_2] 순서를 복원한다.
        for reward in reversed(rewards):
            # 보상을 50으로 나눠 G 와 V 의 스케일을 맞춘다.
            # 예: 마지막 r=30 → G=30/50=0.6. 그 앞 r=20, γ=0.99 → G=20/50 + 0.99*0.6
            discounted = reward / self.reward_scale + self.gamma * discounted
            returns.insert(0, discounted)  # 맨 앞에 넣어 시간 순서를 되돌린다.
        # [2단계] 시간 순서대로 기록한 30개의 경험을 학습용 Tensor로 묶는다.
        # T=30일 때 returns_t, logp, value_t, entropy는 모두 (30,) 형태가 된다.
        # returns_t는 환경 보상에서 직접 계산된 학습 목표이므로 미분 대상이 아니다.
        # log_probs와 entropies는 각 원소 (1,) → cat으로 연결하면 (30,).
        # values는 각 원소 스칼라 → stack으로 새 차원을 만들면 (30,).
        # cat/stack은 기존 Tensor의 자동 미분 그래프를 유지한다.
        # torch.tensor(log_probs)처럼 새로 만들면 그 경사 연결이 끊길 수 있다.
        returns_t = torch.tensor(returns, dtype=torch.float32, device=self.device)
        logp = torch.cat(log_probs)
        value_t = torch.stack(values)
        entropy = torch.cat(entropies)

        # V 는 기준선일 뿐이라 정책 쪽으로 기울기가 흘러가면 안 된다. detach.
        # [3단계] Advantage A_t = 실제 Return G_t − 예상 Return V(s_t).
        # 예: G_t=5, V(s_t)=3 → A_t=+2. 기대보다 잘한 행동이다.
        # 반대로 G_t=1, V(s_t)=3 → A_t=-2. 기대보다 못한 행동이다.
        # 상태 가치 기준선을 빼서 절대 보상 크기보다 해당 상태에서의 상대적 성과를 평가한다.
        # detach()는 V의 숫자를 그대로 쓰되, 정책 손실에서는 이를 상수로 취급한다.
        # 정책 손실이 V 자체를 변경해 Advantage를 조작하는 경로를 차단하는 것이다.
        # 원래 value_t는 여전히 미분 가능하며 아래 value_loss에 사용된다.
        # 정책과 가치 함수가 공유하는 backbone은 각 손실의 경사를 모두 받는다.
        advantages = returns_t - value_t.detach()
        # [4단계] 정책 손실 L_policy = -mean(log π(a_t|s_t) × A_t).
        # A_t>0이면 선택했던 행동의 로그확률밀도를 높이는 방향으로,
        # A_t<0이면 낮추는 방향으로 경사가 계산된다.
        # 예: A=2, logπ=-3일 때 loss=6; logπ=-2로 높아지면 loss=4로 감소한다.
        # 예: A=-2, logπ=-3일 때 loss=-6; logπ=-4로 낮아지면 loss=-8로 감소한다.
        # 연속 분포의 로그확률밀도는 반드시 음수인 것은 아니다(위 숫자는 설명용).
        # 앞의 - 부호는 수업의 경사상승을 PyTorch 옵티마이저의 경사하강으로 구현하기 위함.
        # mean()은 시간별 손실을 평균하여 스칼라로 만든다.
        # 현재 코드는 Return 내부에 γ를 적용하며 손실에 별도 γ^t를 곱하지 않는다.
        policy_loss = -(logp * advantages).mean()
        # 가치 함수는 G 를 맞추도록 따로 학습한다. 맞춰야 A 가 의미 있다.
        # [5단계] 가치 함수 손실 L_value = mean((V(s_t)−G_t)²).
        # 예: 실제 G=5, 예측 V=3이면 제곱오차는 4.
        # V(s)가 실제 Monte Carlo Return에 가까워지도록 학습하는 회귀 문제다.
        # 정책 손실에서 detach한 값과 달리 여기서는 value_t로 경사가 전달된다.
        # TD Actor-Critic과 달리 목표값으로 r+γV(s')가 아닌 실제 G_t를 사용한다.
        value_loss = torch.nn.functional.mse_loss(value_t, returns_t)
        # entropy 를 크게 유지하면 발주가 한 값에 고정되는 것을 늦춘다.
        # [6단계] 탐험을 유지하기 위한 엔트로피 손실 L_entropy = -mean(H).
        # 엔트로피가 커지면 이 손실은 작아지므로, 전체 손실 최소화는
        # 정책이 지나치게 좁은 분포로 수렴하는 것을 완화하는 방향으로 작용한다.
        # 단, 정책 손실과 함께 학습하므로 σ가 항상 증가한다는 뜻은 아니다.
        entropy_loss = -entropy.mean()
        # [7단계] 총 손실 = 정책 손실 + 0.5 × 가치 손실 − 0.01 × 평균 엔트로피.
        # 0.5와 0.01은 각 손실의 영향력을 조정하는 가중치다(기본 설정).
        # 정책 손실 → mean_head, log_std, backbone.
        # 가치 손실 → value_head, backbone.
        # 엔트로피 손실 → log_std(이 구현의 σ는 상태와 독립).
        # 세 손실을 합친 뒤 역전파하므로 한 번의 Adam 업데이트로 함께 학습한다.
        loss = policy_loss + 0.5 * value_loss + self.entropy_coef * entropy_loss

        # [8단계] 경사를 구하고 신경망 파라미터를 실제로 업데이트한다.
        # PyTorch는 기본적으로 경사를 누적하므로 이전 에피소드의 경사를 먼저 지운다.
        self.optimizer.zero_grad()  # 이전 기울기 지우기
        # backward()는 자동 미분 그래프를 거슬러 올라가 각 파라미터의 편미분을 구한다.
        # 이때 파라미터의 값은 아직 바뀌지 않고 .grad에 경사만 기록된다.
        loss.backward()
        # 전체 경사 벡터의 norm이 1을 초과할 경우 축소해 과도한 갱신을 방지한다.
        # 개별 파라미터나 발주량을 1로 제한하는 것이 아니다.
        nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)  # 기울기가 너무 크면 자름
        # Adam이 앞에서 계산한 경사와 학습률 등을 사용하여 W,b,log_std를 갱신한다.
        # 여기서 실제 학습이 발생한다. 30기간마다 한 번 호출하는 것이 핵심이다.
        # 다음 에피소드에서는 갱신된 가우시안 정책으로 다시 발주량을 탐험한다.
        self.optimizer.step()
        # .item()은 스칼라 Tensor의 숫자만 꺼내 학습 로그에 기록하기 위한 것이다.
        # return은 학습용 G_t와 달리 원본 보상을 단순 합산한 에피소드 누적 이익이다.
        return {
            "loss": float(loss.item()),
            "policy_loss": float(policy_loss.item()),
            "value_loss": float(value_loss.item()),
            "return": float(sum(rewards)),  # 로그에는 나누기 전 원래 보상 합
        }

    def save(self, path: str | Path) -> None:
        # 학습이 끝난 모델의 state_dict에는 신경망 가중치, 편향, log_std가 포함된다.
        # 여기서는 Adam 상태나 에피소드 번호를 저장하지 않으므로 완전한 학습 재개용은 아니다.
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model": self.model.state_dict()}, path)  # 가중치만 저장

    def load(self, path: str | Path) -> None:
        # 체크포인트를 현재 연산 장치에 불러온 뒤 기존 신경망에 학습 파라미터를 복원한다.
        # 네트워크 구조는 저장 시와 일치해야 한다. 평가 스크립트가 이 함수를 호출한다.
        payload = torch.load(path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(payload["model"])
