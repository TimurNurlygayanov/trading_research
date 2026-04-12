# Математика для алготрейдинга

**Практическое руководство для CS-инженера: теория, инструменты, границы применимости**

*Эта книга — не учебник математики. Это карта: какие области математики реально используются в алготрейдинге, как они связаны между собой, что можно посчитать руками/библиотеками/LLM, и где находятся границы каждого метода.*

---

## Содержание

1. [Теория вероятностей](#1-теория-вероятностей)
2. [Статистика для трейдера](#2-статистика-для-трейдера)
3. [Временные ряды](#3-временные-ряды)
4. [Стохастические процессы и стохастическое исчисление](#4-стохастические-процессы-и-стохастическое-исчисление)
5. [Линейная алгебра и оптимизация](#5-линейная-алгебра-и-оптимизация)
6. [Теория игр и микроструктура рынка](#6-теория-игр-и-микроструктура-рынка)
7. [Риск-менеджмент](#7-риск-менеджмент)
8. [Машинное обучение в трейдинге](#8-машинное-обучение-в-трейдинге)
9. [Информационная теория](#9-информационная-теория)
10. [Численные методы и Монте-Карло](#10-численные-методы-и-монте-карло)
11. [Инструменты и библиотеки](#11-инструменты-и-библиотеки)
12. [Чеклисты, шпаргалки и war stories](#12-чеклисты-шпаргалки-и-war-stories)

---

## 1. Теория вероятностей

### 1.1 Зачем трейдеру вероятности?

Каждая сделка — это ставка на будущее при неполной информации. Алготрейдинговая стратегия — это машина, которая ищет ситуации с положительным математическим ожиданием и эксплуатирует их систематически.

Три уровня понимания вероятностей в трейдинге:
1. **Базовый:** win rate, expected value, распределения — достаточно для простых стратегий
2. **Продвинутый:** условные вероятности, байесовское обновление, экстремальные распределения — для серьёзного риск-менеджмента
3. **Исследовательский:** мартингалы, мера риск-нейтральности, копулы — для деривативов и глубокого моделирования

### 1.2 Пространство вероятностей (формальный фундамент)

Для CS-инженера полезно знать формальную конструкцию, даже если в повседневной работе она не нужна. Это помогает понять, *почему* некоторые вещи работают, а некоторые — нет.

**Тройка Колмогорова (Ω, F, P):**
- **Ω** — пространство элементарных исходов (все возможные состояния рынка)
- **F** — σ-алгебра событий (набор подмножеств Ω, которым можно присвоить вероятность)
- **P** — вероятностная мера (функция F → [0, 1])

**Зачем это нужно на практике:**
- σ-алгебра формализует понятие "доступная информация". В момент времени t мы знаем F_t (историю до t), но не F_{t+1}. Это **фильтрация** — математическое описание того, что информация постепенно раскрывается
- Понимание фильтраций критично для: определения look-ahead bias в бэктестах, понимания мартингалов, модели ценообразования деривативов

**Случайная величина** — это измеримая функция X: Ω → ℝ. Цена акции в момент t — это случайная величина S_t.

### 1.3 Распределения вероятностей: полная картина

#### Нормальное распределение и почему оно не работает

Нормальное распределение N(μ, σ²) — отправная точка, но финансовые данные систематически от него отклоняются.

**Количественная иллюстрация:**

```python
import numpy as np
from scipy import stats
import yfinance as yf

# Загружаем реальные данные
spy = yf.download('SPY', start='2000-01-01', end='2024-01-01')
returns = spy['Adj Close'].pct_change().dropna()

# Сравниваем хвосты: нормальное vs реальность
sigma = returns.std()
mu = returns.mean()

# Сколько раз за 24 года мы видели движения > 3σ, 4σ, 5σ?
for n_sigma in [3, 4, 5, 6]:
    threshold = n_sigma * sigma
    actual_count = ((returns > threshold) | (returns < -threshold)).sum()
    # Ожидание по нормальному распределению
    expected_prob = 2 * (1 - stats.norm.cdf(n_sigma))
    expected_count = expected_prob * len(returns)
    ratio = actual_count / expected_count if expected_count > 0 else float('inf')
    print(f"{n_sigma}σ: реально={actual_count}, ожидание по N={expected_count:.1f}, "
          f"отношение={ratio:.1f}x")

# Типичный результат:
# 3σ: реально=59, ожидание=16.2, отношение=3.6x
# 4σ: реально=18, ожидание=0.4, отношение=47.7x
# 5σ: реально=8, ожидание=0.003, отношение=2334x
# 6σ: реально=3, ожидание=~0, отношение=∞
```

Событие 5σ по нормальному распределению должно происходить раз в ~14000 лет. В реальности — несколько раз за 20 лет. Это не "чёрный лебедь" — это **системное свойство** финансовых рынков.

#### Семейство распределений для финансов

**Распределение Стьюдента (t-distribution):**
- Параметр ν (степени свободы) контролирует толщину хвостов
- ν = 3-5 хорошо описывает дневные доходности акций
- При ν ≤ 2 дисперсия бесконечна — ЦПТ не работает!
- Обобщение: **skewed t-distribution** (добавляет асимметрию)

```python
from scipy.stats import t as t_dist

# Подбор параметров t-распределения к реальным данным
params = t_dist.fit(returns)
nu, loc, scale = params
print(f"Степени свободы ν: {nu:.2f}")  # типично 3-6 для акций
print(f"Если ν < 4, эксцесс бесконечен — осторожно с оценками!")

# Сравнение хвостов
x = np.linspace(-0.05, 0.05, 1000)
plt.plot(x, stats.norm.pdf(x, mu, sigma), label='Normal', linestyle='--')
plt.plot(x, t_dist.pdf(x, *params), label=f't (ν={nu:.1f})')
plt.hist(returns, bins=100, density=True, alpha=0.3, label='Реальные')
plt.legend()
plt.title('Нормальное vs t vs реальность')
plt.yscale('log')  # Лог-шкала — чтобы видеть хвосты
plt.show()
```

**Generalized Pareto Distribution (GPD) — для хвостов:**
- Extreme Value Theory (EVT) — отдельная область математики, специально для моделирования редких событий
- Суть: отдельно моделируешь "тело" и "хвосты" распределения
- Теорема Пикандса-Балкемы-де Хаана: для достаточно больших порогов хвосты *любого* распределения описываются GPD

```python
from scipy.stats import genpareto

# Берём только экстремальные отрицательные доходности (ниже 5-го перцентиля)
threshold = returns.quantile(0.05)
tail_data = -(returns[returns < threshold] - threshold)  # превышения, положительные

# Подбираем GPD
shape, loc, scale = genpareto.fit(tail_data, floc=0)
print(f"Shape parameter ξ: {shape:.3f}")
# ξ > 0: тяжёлые хвосты (типично для финансов)
# ξ = 0: экспоненциальные хвосты
# ξ < 0: ограниченные хвосты

# Оценка VaR на уровне 99.9% через EVT (намного точнее, чем нормальное)
p = 0.001
n_exceed = len(tail_data)
n_total = len(returns)
exceed_prob = n_exceed / n_total
var_evt = threshold - scale/shape * ((p / exceed_prob)**(-shape) - 1)
print(f"VaR 99.9% (EVT): {var_evt:.4f}")
print(f"VaR 99.9% (Normal): {stats.norm.ppf(0.001, mu, sigma):.4f}")
```

**Stable distributions (распределения Леви):**
- Обобщение нормального: параметр α ∈ (0, 2] контролирует хвосты
- α = 2 → нормальное, α < 2 → бесконечная дисперсия
- Mandelbrot предлагал α ≈ 1.7 для финансов
- На практике редко используются напрямую, но важны концептуально: если α < 2, то выборочная дисперсия не сходится — все стандартные статистики (Sharpe, VaR) нестабильны

```python
from scipy.stats import levy_stable

# Подбор стабильного распределения (может быть медленным)
params = levy_stable.fit(returns)
alpha, beta, loc, scale = params
print(f"α (индекс устойчивости): {alpha:.3f}")  # < 2 = толстые хвосты
print(f"β (асимметрия): {beta:.3f}")  # -1..1
```

**Смесь нормальных (Gaussian Mixture):**
- Практичная альтернатива: 2-3 нормальных распределения, смешанных с весами
- Интерпретация: разные "режимы" рынка (спокойный, волатильный, кризис)

```python
from sklearn.mixture import GaussianMixture

gmm = GaussianMixture(n_components=3, random_state=42)
gmm.fit(returns.values.reshape(-1, 1))

for i in range(3):
    w = gmm.weights_[i]
    m = gmm.means_[i][0]
    s = np.sqrt(gmm.covariances_[i][0][0])
    print(f"Режим {i}: вес={w:.2%}, μ={m:.5f}, σ={s:.5f}")
# Типично: ~80% спокойный режим, ~15% повышенная vol, ~5% кризис
```

### 1.4 Математическое ожидание: больше, чем среднее

#### Закон полного ожидания (Tower property)

```
E[E[X|Y]] = E[X]
```

**Для трейдинга:** если знаешь ожидаемую доходность в каждом режиме рынка, общая ожидаемая доходность — взвешенное среднее:

```python
# E[return] = P(bull)*E[return|bull] + P(bear)*E[return|bear] + P(crisis)*E[return|crisis]

regimes = {
    'bull':   {'prob': 0.60, 'mean_return': 0.0005, 'vol': 0.008},
    'bear':   {'prob': 0.30, 'mean_return': -0.0002, 'vol': 0.015},
    'crisis': {'prob': 0.10, 'mean_return': -0.003, 'vol': 0.04},
}

total_expected = sum(r['prob'] * r['mean_return'] for r in regimes.values())
total_vol = np.sqrt(sum(
    r['prob'] * (r['vol']**2 + r['mean_return']**2) for r in regimes.values()
) - total_expected**2)

print(f"Общее E[r] = {total_expected:.6f}/день = {total_expected*252:.4f}/год")
print(f"Общая vol = {total_vol:.6f}/день = {total_vol*np.sqrt(252):.4f}/год")
```

#### Conditional expectation и предсказание

Условное ожидание E[Y|X] — лучший (в смысле MSE) предсказатель Y по X. Любая предсказательная модель — это попытка оценить условное ожидание.

Линейная регрессия оценивает E[Y|X] = βX + α, предполагая линейность. Нейронная сеть оценивает E[Y|X] = f(X) для произвольной f. Но если E[Y|X] = E[Y] (Y не зависит от X), никакая модель не поможет — и это часто случай для финансовых доходностей.

#### Jensen's inequality

Если f — выпуклая функция, то E[f(X)] ≥ f(E[X]).

**Применение в трейдинге:**
- Волатильность вредит компаундингу: E[log(1+r)] < log(1+E[r]). Два портфеля с одинаковой средней доходностью, но разной волатильностью будут расти по-разному — менее волатильный вырастет больше. Это **volatility drag**

```python
# Демонстрация volatility drag
np.random.seed(42)
n_years = 20

# Стратегия A: 10% годовых, 15% vol
# Стратегия B: 10% годовых, 30% vol
for name, vol in [('A (low vol)', 0.15), ('B (high vol)', 0.30)]:
    annual_return = 0.10
    daily_return = annual_return / 252
    daily_vol = vol / np.sqrt(252)

    # Симулируем 1000 путей
    paths = np.zeros((1000, 252 * n_years))
    for i in range(1000):
        daily_r = np.random.normal(daily_return, daily_vol, 252 * n_years)
        paths[i] = np.cumprod(1 + daily_r)

    median_final = np.median(paths[:, -1])
    mean_final = np.mean(paths[:, -1])
    # Геометрическая средняя годовая доходность
    geo_return = median_final ** (1/n_years) - 1
    print(f"{name}: median final=${median_final:.2f}, geo annual={geo_return:.2%}")
    # B будет значительно хуже несмотря на одинаковое E[r]

# Формула: geo_return ≈ arith_return - σ²/2
for vol in [0.15, 0.30, 0.50]:
    drag = vol**2 / 2
    print(f"Vol={vol:.0%}: volatility drag = {drag:.2%}/год")
```

### 1.5 Теорема Байеса: обновление убеждений

#### Байесовский фреймворк для оценки стратегий

Классический вопрос: стратегия показала Sharpe = 1.5 на бэктесте. Какова вероятность, что она реально рабочая?

```python
import pymc as pm
import arviz as az

# Байесовская оценка Sharpe Ratio
# Вместо точечной оценки получаем РАСПРЕДЕЛЕНИЕ Sharpe
with pm.Model() as model:
    # Априорные распределения
    mu = pm.Normal('mu', mu=0, sigma=0.001)        # слабый prior на среднюю
    sigma = pm.HalfNormal('sigma', sigma=0.02)      # prior на vol

    # Likelihood (t-распределение — устойчиво к выбросам)
    nu = pm.Exponential('nu', 1/30) + 2  # степени свободы > 2
    returns_obs = pm.StudentT('returns', nu=nu, mu=mu, sigma=sigma,
                               observed=returns_data)

    # MCMC семплирование
    trace = pm.sample(2000, return_inferencedata=True)

# Posterior Sharpe Ratio
sharpe_samples = trace.posterior['mu'].values.flatten() / trace.posterior['sigma'].values.flatten() * np.sqrt(252)
print(f"Sharpe: {np.mean(sharpe_samples):.2f} "
      f"[{np.percentile(sharpe_samples, 2.5):.2f}, {np.percentile(sharpe_samples, 97.5):.2f}]")
# Получаем доверительный интервал, а не точку!

# P(Sharpe > 0)?
prob_positive = (sharpe_samples > 0).mean()
print(f"P(Sharpe > 0) = {prob_positive:.2%}")
```

#### Байесовское A/B тестирование стратегий

```python
# Какая стратегия лучше? Не "p < 0.05", а "вероятность, что A лучше B"
with pm.Model() as ab_model:
    mu_a = pm.Normal('mu_a', mu=0, sigma=0.005)
    mu_b = pm.Normal('mu_b', mu=0, sigma=0.005)
    sigma_a = pm.HalfNormal('sigma_a', sigma=0.02)
    sigma_b = pm.HalfNormal('sigma_b', sigma=0.02)

    pm.Normal('obs_a', mu=mu_a, sigma=sigma_a, observed=returns_a)
    pm.Normal('obs_b', mu=mu_b, sigma=sigma_b, observed=returns_b)

    # Derived: разница
    diff = pm.Deterministic('diff', mu_a - mu_b)

    trace = pm.sample(2000, return_inferencedata=True)

# P(стратегия A лучше B)
diff_samples = trace.posterior['diff'].values.flatten()
p_a_better = (diff_samples > 0).mean()
print(f"P(A лучше B) = {p_a_better:.2%}")
```

#### Conjugate priors — когда можно считать аналитически

Для некоторых комбинаций prior/likelihood есть точные формулы (без MCMC):

| Prior | Likelihood | Posterior | Применение |
|-------|-----------|-----------|------------|
| Beta(α,β) | Bernoulli | Beta(α+wins, β+losses) | Win rate |
| Normal | Normal | Normal | Средняя доходность |
| Gamma | Poisson | Gamma | Частота событий |
| Normal-InvGamma | Normal (неизв. μ,σ) | Normal-InvGamma | Доходность с неизв. vol |

```python
from scipy.stats import beta

# Пример: обновление оценки win rate
# Prior: Beta(2, 2) — слабое убеждение, что win rate ≈ 50%
alpha_prior, beta_prior = 2, 2

# Данные: 60 побед из 100 сделок
wins, losses = 60, 40

# Posterior: Beta(62, 42)
alpha_post = alpha_prior + wins
beta_post = beta_prior + losses

x = np.linspace(0, 1, 1000)
plt.plot(x, beta.pdf(x, alpha_prior, beta_prior), label='Prior', linestyle='--')
plt.plot(x, beta.pdf(x, alpha_post, beta_post), label='Posterior')
plt.axvline(wins/(wins+losses), color='red', linestyle=':', label='MLE')
plt.legend()
plt.title('Байесовская оценка win rate')
plt.show()

# 95% credible interval
ci = beta.interval(0.95, alpha_post, beta_post)
print(f"Win rate: {alpha_post/(alpha_post+beta_post):.2%} [{ci[0]:.2%}, {ci[1]:.2%}]")
```

### 1.6 Копулы: зависимость за пределами корреляции

Корреляция измеряет только *линейную* зависимость и полностью определяет совместное распределение только для многомерного нормального. Но финансовые активы имеют сложную структуру зависимости, особенно в хвостах (в кризис всё коррелирует сильнее).

**Копула** — функция, которая связывает маргинальные распределения в совместное. По теореме Склара, любое совместное распределение можно разложить: маргиналы + копула.

```python
from scipy.stats import kendalltau, norm
# Для полноценной работы с копулами: pip install copulas

# Типы копул:
# - Гауссова: зависимость как у многомерного нормального (без tail dependence)
# - t-копула: допускает tail dependence (когда экстремы происходят вместе)
# - Clayton: нижняя tail dependence (совместные обвалы)
# - Gumbel: верхняя tail dependence (совместные ралли)
# - Frank: симметричная, без tail dependence

# Tail dependence coefficient — ключевая метрика
# "Если актив A упал ниже 1-го перцентиля, какова вероятность, что B тоже?"
def empirical_tail_dependence(returns_a, returns_b, quantile=0.05):
    """Эмпирический коэффициент хвостовой зависимости."""
    threshold_a = returns_a.quantile(quantile)
    threshold_b = returns_b.quantile(quantile)
    both_extreme = ((returns_a < threshold_a) & (returns_b < threshold_b)).mean()
    return both_extreme / quantile

# Сравниваем: обычная корреляция vs хвостовая зависимость
corr = returns_a.corr(returns_b)
tail_dep = empirical_tail_dependence(returns_a, returns_b)
print(f"Корреляция: {corr:.3f}")
print(f"Хвостовая зависимость (5%): {tail_dep:.3f}")
# Часто tail_dep >> корr — диверсификация исчезает в кризис
```

**Зачем копулы:**
- Стресс-тестирование портфеля (корреляции в кризис ≠ корреляции в спокойное время)
- Оценка CDO и структурированных продуктов (именно неправильная копула привела к кризису 2008 — все использовали Гауссову, которая недооценивала совместные дефолты)
- Multi-asset стратегии: правильная оценка совместных рисков

### 1.7 Мартингалы

**Мартингал** — процесс, для которого лучший прогноз будущего значения — текущее значение:

```
E[X_{t+1} | F_t] = X_t
```

**Почему это важно:**
1. **Гипотеза эффективного рынка (EMH)** утверждает, что цены (с поправкой на дрифт) — мартингал. Если это так, никакая стратегия не может систематически зарабатывать (кроме risk premium)
2. **Risk-neutral pricing**: в риск-нейтральной мере дисконтированные цены — мартингал. На этом стоит вся теория ценообразования деривативов
3. **Optional stopping theorem**: нельзя обыграть мартингал стоп-лоссом или тейк-профитом. Если E[r] = 0, то E[r со стопами] = 0 (при определённых условиях)

```python
# Проверка: является ли ряд мартингалом?
# Простой тест: автокорреляция приращений должна быть ~0

from statsmodels.stats.diagnostic import acorr_ljungbox

# Тест Льюнга-Бокса на автокорреляцию
lb_test = acorr_ljungbox(returns, lags=20, return_df=True)
print(lb_test)
# Если p-values > 0.05 для всех лагов — совместимо с мартингалом

# Variance ratio test — более мощный тест
from statsmodels.stats.diagnostic import het_arch

# Если отношение var(r_k) / (k * var(r_1)) ≠ 1, ряд — не random walk
def variance_ratio_test(returns, k=5):
    """Variance ratio test (Lo-MacKinlay)."""
    n = len(returns)
    var_1 = returns.var()
    # k-периодные доходности
    returns_k = returns.rolling(k).sum().dropna()
    var_k = returns_k.var()

    vr = var_k / (k * var_1)
    # Под H0 (random walk): VR = 1
    # Статистика
    se = np.sqrt(2 * (2*k - 1) * (k - 1) / (3 * k * n))
    z = (vr - 1) / se
    p_value = 2 * (1 - stats.norm.cdf(abs(z)))

    print(f"Variance Ratio ({k}): {vr:.4f}")
    print(f"z-статистика: {z:.3f}, p-value: {p_value:.4f}")
    if vr > 1:
        print("VR > 1 → тренд (positive autocorrelation)")
    else:
        print("VR < 1 → mean-reversion (negative autocorrelation)")
    return vr, p_value

for k in [2, 5, 10, 20]:
    variance_ratio_test(returns, k)
    print()
```

### 1.8 Закон больших чисел и скорость сходимости

ЗБЧ говорит, что выборочное среднее сходится к истинному. Но **как быстро**?

```python
# Для нормального распределения: стандартная ошибка = σ/√n
# Доверительный интервал для среднего: x̄ ± z * σ/√n

def required_trades(target_precision, daily_vol, confidence=0.95):
    """Сколько сделок нужно для оценки средней доходности с заданной точностью?"""
    z = stats.norm.ppf(1 - (1 - confidence) / 2)
    n = (z * daily_vol / target_precision) ** 2
    return int(np.ceil(n))

# Хотим оценить среднюю дневную доходность с точностью ±0.01% (0.0001)
# При типичной дневной vol = 1% (0.01)
n = required_trades(0.0001, 0.01)
print(f"Нужно сделок: {n}")  # ≈ 38416!
print(f"Это ≈ {n/252:.0f} лет ежедневной торговли")
# Вот почему статистическая оценка стратегий так трудна
```

**Для толстых хвостов ситуация хуже:** если распределение имеет бесконечную дисперсию (α-stable с α < 2), обычный ЗБЧ не работает. Сходимость идёт по другому закону, и выборочное среднее очень нестабильно.

### 1.9 Множественное тестирование: проблема #1 в алготрейдинге

Если тестируешь N стратегий, вероятность найти хотя бы одну "значимую" случайно:

```
P(хотя бы 1 ложная) = 1 - (1 - α)^N
```

При N=100 и α=0.05: P = 1 - 0.95^100 ≈ 99.4%

```python
from statsmodels.stats.multitest import multipletests

def deflated_sharpe_ratio(observed_sharpe, n_trials, n_observations,
                          skew=0, kurtosis=3, var_sharpe=None):
    """
    Deflated Sharpe Ratio (Bailey & Lopez de Prado, 2014).
    Корректирует Sharpe за количество протестированных стратегий.

    Это THE метод для оценки стратегий в quant finance.
    """
    if var_sharpe is None:
        # Оценка дисперсии Sharpe
        var_sharpe = (1 + 0.5 * observed_sharpe**2 -
                      skew * observed_sharpe +
                      ((kurtosis - 3) / 4) * observed_sharpe**2) / n_observations

    # Expected maximum Sharpe из n_trials независимых тестов
    # (по формуле для максимума N нормальных)
    e_max = stats.norm.ppf(1 - 1/n_trials) * np.sqrt(var_sharpe)

    # Сколько σ наблюдаемый Sharpe выше E[max]?
    dsr = stats.norm.cdf((observed_sharpe - e_max) / np.sqrt(var_sharpe))
    return dsr

# Пример: Sharpe = 2.0, но тестировали 500 стратегий
dsr = deflated_sharpe_ratio(
    observed_sharpe=2.0,
    n_trials=500,
    n_observations=252 * 5,  # 5 лет дневных данных
    skew=-0.5,
    kurtosis=5
)
print(f"Deflated Sharpe Ratio p-value: {dsr:.4f}")
# Может оказаться незначимым!
```

### 1.10 Границы и подводные камни — расширенная таблица

| Проблема | Описание | Серьёзность | Решение | Инструмент |
|----------|----------|-------------|---------|------------|
| Fat tails | Экстремы чаще, чем в модели | Критическая | EVT, GPD, t-distribution | `scipy.stats.genpareto` |
| Нестационарность | Параметры меняются со временем | Критическая | Скользящие окна, HMM | `hmmlearn`, `ruptures` |
| Множественное тестирование | Ложные сигналы при N тестах | Критическая | Deflated Sharpe, FDR | `statsmodels.multitest` |
| Tail dependence | В кризис все корреляции → 1 | Высокая | Копулы, стресс-тесты | `copulas` |
| Autocorrelation в vol | Волатильность кластеризуется | Средняя | GARCH, HAR | `arch` |
| Микроструктурный шум | Tick-данные содержат шум | Средняя для HFT | Фильтрация, агрегация | - |
| Survivorship bias | Данные без мёртвых компаний | Высокая | Delisted-inclusive данные | - |
| Non-synchronous trading | Активы торгуются в разное время | Средняя | Hayashi-Yoshida estimator | - |

---

## 2. Статистика для трейдера

### 2.1 Описательная статистика: полный набор

```python
import pandas as pd
import numpy as np
from scipy import stats

def full_trading_stats(returns: pd.Series) -> pd.Series:
    """Расширенный набор статистик."""
    n = len(returns)
    result = {
        # Центральная тенденция
        'Средняя доходность (дн.)': returns.mean(),
        'Средняя доходность (год.)': returns.mean() * 252,
        'Медиана': returns.median(),
        'Geometric mean (дн.)': (1 + returns).prod() ** (1/n) - 1,

        # Разброс
        'Std (дн.)': returns.std(),
        'Std (год.)': returns.std() * np.sqrt(252),
        'MAD (median abs deviation)': stats.median_abs_deviation(returns),
        'IQR': returns.quantile(0.75) - returns.quantile(0.25),

        # Форма распределения
        'Skewness': returns.skew(),
        'Kurtosis (excess)': returns.kurtosis(),
        'Jarque-Bera p-value': stats.jarque_bera(returns)[1],

        # Хвосты
        'VaR 1%': returns.quantile(0.01),
        'VaR 5%': returns.quantile(0.05),
        'CVaR 1%': returns[returns <= returns.quantile(0.01)].mean(),
        'CVaR 5%': returns[returns <= returns.quantile(0.05)].mean(),
        'Max loss (1 day)': returns.min(),
        'Max gain (1 day)': returns.max(),

        # Серийная зависимость
        'Autocorr lag-1': returns.autocorr(1),
        'Autocorr lag-5': returns.autocorr(5),
        'Ljung-Box p (lag 10)': acorr_ljungbox(returns, lags=10, return_df=True)['lb_pvalue'].iloc[-1],

        # Производительность
        'Sharpe': returns.mean() / returns.std() * np.sqrt(252),
        'Sortino': returns.mean() / returns[returns < 0].std() * np.sqrt(252),
        'Win rate': (returns > 0).mean(),
        'Profit factor': returns[returns > 0].sum() / abs(returns[returns < 0].sum()),
        'Avg win / avg loss': returns[returns > 0].mean() / abs(returns[returns < 0].mean()),
    }
    return pd.Series(result)
```

### 2.2 Проверка гипотез: глубже

#### Power analysis — планирование эксперимента

**Мощность теста** — вероятность обнаружить эффект, если он есть. Стандарт — 80%.

```python
from statsmodels.stats.power import TTestPower

power_analysis = TTestPower()

# Сколько наблюдений нужно, чтобы обнаружить Sharpe = 0.5?
# effect_size для t-теста = mean / std = Sharpe / sqrt(252) (для дневных данных)
sharpe_target = 0.5
effect_size = sharpe_target / np.sqrt(252)  # ≈ 0.0315

n_required = power_analysis.solve_power(
    effect_size=effect_size,
    power=0.80,
    alpha=0.05
)
print(f"Для обнаружения Sharpe={sharpe_target}: {n_required:.0f} дней ({n_required/252:.1f} лет)")
# Типично: 3-7 лет для Sharpe = 0.5, что объясняет, почему так трудно
# отличить рабочую стратегию от шума

# Sharpe = 1.0?
effect_size_high = 1.0 / np.sqrt(252)
n_high = power_analysis.solve_power(effect_size=effect_size_high, power=0.80, alpha=0.05)
print(f"Для Sharpe={1.0}: {n_high:.0f} дней ({n_high/252:.1f} лет)")

# Sharpe = 2.0?
effect_size_very_high = 2.0 / np.sqrt(252)
n_very_high = power_analysis.solve_power(effect_size=effect_size_very_high, power=0.80, alpha=0.05)
print(f"Для Sharpe={2.0}: {n_very_high:.0f} дней ({n_very_high/252:.1f} лет)")
```

**Ключевой вывод:** нужно 3-5 лет данных, чтобы отличить Sharpe=0.5 от шума с 80% уверенностью. Для стратегий с более низким Sharpe — ещё больше. Вот почему стратегии часто "работают" на бэктесте и "ломаются" на живых данных: недостаточно данных для надёжной оценки.

#### Permutation tests (рандомизационные тесты)

Не требуют никаких предположений о распределении. Идея: перемешиваем данные, разрушая зависимость, и смотрим, мог ли результат возникнуть случайно.

```python
def permutation_test_sharpe(returns, n_permutations=10000):
    """
    Тест: значим ли Sharpe ratio?
    Перемешиваем порядок доходностей — если порядок не важен,
    Sharpe не должен измениться. Для iid данных он не изменится,
    но если стратегия зависит от порядка (momentum, mean-reversion),
    перемешивание уничтожит сигнал.
    """
    observed_sharpe = returns.mean() / returns.std() * np.sqrt(252)

    perm_sharpes = []
    for _ in range(n_permutations):
        perm = np.random.permutation(returns.values)
        perm_sharpe = perm.mean() / perm.std() * np.sqrt(252)
        perm_sharpes.append(perm_sharpe)

    p_value = np.mean(np.array(perm_sharpes) >= observed_sharpe)
    return observed_sharpe, p_value, perm_sharpes

sharpe, p, perm_dist = permutation_test_sharpe(strategy_returns)
print(f"Observed Sharpe: {sharpe:.3f}, p-value: {p:.4f}")

# Визуализация
plt.hist(perm_dist, bins=50, alpha=0.7, label='Permuted')
plt.axvline(sharpe, color='red', linewidth=2, label=f'Observed (p={p:.3f})')
plt.legend()
plt.title('Permutation test for Sharpe Ratio')
plt.show()
```

#### Bootstrap: универсальный швейцарский нож

```python
def bootstrap_ci(data, statistic_fn, n_bootstrap=10000, ci=0.95):
    """
    Bootstrap доверительный интервал для ЛЮБОЙ статистики.
    statistic_fn — функция, принимающая массив и возвращающая число.
    """
    boot_stats = []
    n = len(data)
    for _ in range(n_bootstrap):
        sample = np.random.choice(data, size=n, replace=True)
        boot_stats.append(statistic_fn(sample))

    boot_stats = np.array(boot_stats)
    alpha = (1 - ci) / 2
    lower = np.percentile(boot_stats, alpha * 100)
    upper = np.percentile(boot_stats, (1 - alpha) * 100)

    return np.mean(boot_stats), lower, upper

# Примеры использования
mean, lo, hi = bootstrap_ci(returns.values, np.mean)
print(f"Mean return: {mean:.6f} [{lo:.6f}, {hi:.6f}]")

# Sharpe ratio CI
sharpe_fn = lambda r: np.mean(r) / np.std(r) * np.sqrt(252)
sharpe, lo, hi = bootstrap_ci(returns.values, sharpe_fn)
print(f"Sharpe: {sharpe:.3f} [{lo:.3f}, {hi:.3f}]")

# Max drawdown CI
def max_dd(r):
    equity = np.cumprod(1 + r)
    peak = np.maximum.accumulate(equity)
    dd = (equity - peak) / peak
    return dd.min()

mdd, lo, hi = bootstrap_ci(returns.values, max_dd)
print(f"Max DD: {mdd:.2%} [{lo:.2%}, {hi:.2%}]")
```

**Block bootstrap** — для временных рядов (сохраняет автокорреляцию):

```python
from arch.bootstrap import StationaryBootstrap

# Стационарный bootstrap — длина блоков случайная (геометрическое распределение)
bs = StationaryBootstrap(20, returns.values)  # средняя длина блока = 20

boot_sharpes = []
for data in bs.bootstrap(1000):
    r = data[0][0]
    boot_sharpes.append(r.mean() / r.std() * np.sqrt(252))

print(f"Block bootstrap Sharpe: {np.mean(boot_sharpes):.3f} "
      f"[{np.percentile(boot_sharpes, 2.5):.3f}, {np.percentile(boot_sharpes, 97.5):.3f}]")
```

### 2.3 Корреляция: вся правда

#### Три вида корреляции

```python
# 1. Pearson — линейная зависимость. Чувствительна к выбросам
pearson = returns_a.corr(returns_b)

# 2. Spearman — ранговая (монотонная). Устойчива к выбросам
spearman, _ = stats.spearmanr(returns_a, returns_b)

# 3. Kendall — ранговая, но на основе concordance. Ещё более устойчива
kendall, _ = stats.kendalltau(returns_a, returns_b)

print(f"Pearson: {pearson:.4f}")
print(f"Spearman: {spearman:.4f}")
print(f"Kendall: {kendall:.4f}")
# Если Pearson << Spearman — зависимость нелинейна или есть выбросы
```

#### Distance correlation — ловит нелинейные зависимости

В отличие от Pearson, distance correlation = 0 **тогда и только тогда**, когда переменные независимы.

```python
from dcor import distance_correlation

dcor = distance_correlation(returns_a.values, returns_b.values)
print(f"Distance correlation: {dcor:.4f}")
# Может быть > 0, даже если Pearson ≈ 0 (нелинейная зависимость)
```

#### Скользящая корреляция и DCC-GARCH

```python
# Простая скользящая
rolling_corr = returns_a.rolling(60).corr(returns_b)

# Экспоненциально взвешенная (свежие данные важнее)
ewm_corr = returns_a.ewm(span=60).corr(returns_b)

# DCC-GARCH — динамическая условная корреляция
# Корреляция, которая учитывает волатильность и кластеризацию
from arch import arch_model

# Шаг 1: GARCH для каждого ряда
model_a = arch_model(returns_a * 100, vol='Garch', p=1, q=1).fit(disp='off')
model_b = arch_model(returns_b * 100, vol='Garch', p=1, q=1).fit(disp='off')

# Стандартизированные остатки
std_resid_a = model_a.std_resid
std_resid_b = model_b.std_resid

# Шаг 2: DCC на стандартизированных остатках
# (для полного DCC используй rmgarch в R или кастомную реализацию)
# Простое приближение — EWMA на стандартизированных остатках:
dcc_approx = pd.Series(std_resid_a * std_resid_b).ewm(span=60).mean()
```

### 2.4 Регрессия: расширенный арсенал

#### OLS с робастными стандартными ошибками

```python
import statsmodels.api as sm

X = sm.add_constant(factors)

# Обычный OLS — предполагает гомоскедастичность (нереалистично для финансов)
model_ols = sm.OLS(returns, X).fit()

# HC3 — робастные стандартные ошибки (Heteroscedasticity-Consistent)
model_hc3 = sm.OLS(returns, X).fit(cov_type='HC3')

# HAC — Newey-West (учитывает и гетероскедастичность, и автокорреляцию)
model_hac = sm.OLS(returns, X).fit(cov_type='HAC', cov_kwds={'maxlags': 5})

print("OLS SE vs HC3 SE vs HAC SE:")
comparison = pd.DataFrame({
    'OLS': model_ols.bse,
    'HC3': model_hc3.bse,
    'HAC': model_hac.bse
})
print(comparison)
# HAC SE обычно больше → "значимые" факторы могут стать незначимыми
```

#### Quantile regression

OLS оценивает условное **среднее**. Quantile regression оценивает условные **квантили** — как факторы влияют на хвосты?

```python
import statsmodels.formula.api as smf

# Как фактор влияет на 5-й перцентиль доходности (левый хвост)?
model_q05 = smf.quantreg('returns ~ market + volume', data=df).fit(q=0.05)
# Как на 50-й (медиану)?
model_q50 = smf.quantreg('returns ~ market + volume', data=df).fit(q=0.50)
# Как на 95-й (правый хвост)?
model_q95 = smf.quantreg('returns ~ market + volume', data=df).fit(q=0.95)

print("Влияние market на разные квантили:")
print(f"  5%:  {model_q05.params['market']:.4f}")
print(f"  50%: {model_q50.params['market']:.4f}")
print(f"  95%: {model_q95.params['market']:.4f}")
# Если коэффициенты разные — влияние несимметрично!
```

#### LASSO и Ridge для feature selection

```python
from sklearn.linear_model import LassoCV, RidgeCV
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_train)

# LASSO (L1) — обнуляет ненужные коэффициенты → feature selection
lasso = LassoCV(cv=TimeSeriesSplit(5)).fit(X_scaled, y_train)
selected = np.array(feature_names)[lasso.coef_ != 0]
print(f"LASSO выбрал {len(selected)} из {len(feature_names)} фичей: {selected}")

# Ridge (L2) — сжимает коэффициенты, но не обнуляет
ridge = RidgeCV(cv=TimeSeriesSplit(5).split(X_scaled)).fit(X_scaled, y_train)

# Elastic Net — комбинация L1 + L2
from sklearn.linear_model import ElasticNetCV
enet = ElasticNetCV(l1_ratio=[0.1, 0.5, 0.9], cv=TimeSeriesSplit(5)).fit(X_scaled, y_train)
```

### 2.5 Тест Грейнджера: причинность во временных рядах

"X Granger-причиняет Y" = прошлые значения X помогают предсказать Y (сверх прошлых значений Y).

```python
from statsmodels.tsa.stattools import grangercausalitytests

# Проверяем: объём → доходность? доходность → объём?
result = grangercausalitytests(df[['returns', 'volume_change']], maxlag=5, verbose=True)

# Часто обнаруживается:
# - Доходность → объём (да, люди торгуют на движениях)
# - Объём → доходность (слабо или нет — зависит от актива)
```

**Ограничение:** Granger causality — это предсказательная способность, а не истинная причинность. Может быть общий скрытый фактор.

### 2.6 Change-point detection

Обнаружение моментов, когда статистические свойства ряда изменились (смена режима, структурный слом):

```python
import ruptures

# PELT — оптимальный алгоритм для множественных change points
algo = ruptures.Pelt(model="rbf").fit(returns.values)
breakpoints = algo.predict(pen=10)  # pen — штраф за сложность
print(f"Найдено {len(breakpoints)-1} точек перелома: {breakpoints}")

# Визуализация
ruptures.display(returns.values, breakpoints)
plt.title('Change-point detection')
plt.show()

# CUSUM — кумулятивная сумма (классический метод для online detection)
def cusum_test(series, threshold=5):
    """Детектирует сдвиг среднего в реальном времени."""
    mu = series.mean()
    sigma = series.std()
    s_pos, s_neg = 0, 0
    alarms = []
    for i, x in enumerate(series):
        z = (x - mu) / sigma
        s_pos = max(0, s_pos + z - 0.5)
        s_neg = max(0, s_neg - z - 0.5)
        if s_pos > threshold or s_neg > threshold:
            alarms.append(i)
            s_pos, s_neg = 0, 0  # reset
    return alarms
```

---

## 3. Временные ряды

### 3.1 Стационарность: формальное определение и тесты

#### Виды стационарности

- **Строгая стационарность:** совместное распределение (X_t1, ..., X_tk) = совместное распределение (X_{t1+h}, ..., X_{tk+h}) для любого h. На практике непроверяема
- **Слабая (ковариационная) стационарность:** E[X_t] = const, Cov(X_t, X_{t+h}) зависит только от h. Проверяема и достаточна для большинства методов

**Цены → не стационарны. Доходности → слабо стационарны. Лог-доходности → предпочтительны.**

Почему лог-доходности лучше:
- Аддитивны: r(t, t+2) = r(t, t+1) + r(t+1, t+2)
- Симметричны: +5% и -5% дают одинаковый |log-return|
- Не ограничены снизу -100% (обычные — ограничены)

```python
log_returns = np.log(prices / prices.shift(1)).dropna()
# vs
simple_returns = prices.pct_change().dropna()
# Для малых r: log(1+r) ≈ r, разница мала
# Для больших r: отличия существенны
```

#### Расширенный набор тестов на стационарность

```python
from statsmodels.tsa.stattools import adfuller, kpss
from arch.unitroot import PhillipsPerron, ZivotAndrews

def stationarity_tests(series, name='series'):
    """Комплексная проверка стационарности."""
    print(f"=== Тесты стационарности для {name} ===")

    # ADF: H0 = есть единичный корень (нестационарен)
    adf_stat, adf_p, _, _, adf_crit, _ = adfuller(series, autolag='AIC')
    print(f"ADF: stat={adf_stat:.4f}, p={adf_p:.4f} "
          f"{'✓ стационарен' if adf_p < 0.05 else '✗ нестационарен'}")

    # KPSS: H0 = стационарен (обратная H0!)
    kpss_stat, kpss_p, _, kpss_crit = kpss(series, regression='c', nlags='auto')
    print(f"KPSS: stat={kpss_stat:.4f}, p={kpss_p:.4f} "
          f"{'✓ стационарен' if kpss_p > 0.05 else '✗ нестационарен'}")

    # Phillips-Perron: как ADF, но устойчив к автокорреляции
    pp = PhillipsPerron(series)
    print(f"PP: stat={pp.stat:.4f}, p={pp.pvalue:.4f} "
          f"{'✓' if pp.pvalue < 0.05 else '✗'}")

    # Zivot-Andrews: ADF с возможным структурным сломом
    za = ZivotAndrews(series)
    print(f"Zivot-Andrews: stat={za.stat:.4f}, p={za.pvalue:.4f}, "
          f"break date index={za.breakpoint}")

    # Интерпретация
    if adf_p < 0.05 and kpss_p > 0.05:
        print("→ Ряд стационарен (оба теста согласны)")
    elif adf_p >= 0.05 and kpss_p <= 0.05:
        print("→ Ряд нестационарен (оба теста согласны)")
    else:
        print("→ Тесты не согласны — возможно, тренд-стационарен или есть структурный слом")

stationarity_tests(returns, 'SPY returns')
stationarity_tests(prices, 'SPY prices')
```

### 3.2 Автокорреляция: ACF, PACF и их интерпретация

```python
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# ACF/PACF доходностей
plot_acf(returns, lags=40, ax=axes[0,0], title='ACF доходностей')
plot_pacf(returns, lags=40, ax=axes[0,1], title='PACF доходностей')

# ACF/PACF квадратов доходностей (= волатильность)
plot_acf(returns**2, lags=40, ax=axes[1,0], title='ACF |returns|² (волатильность)')
plot_pacf(returns**2, lags=40, ax=axes[1,1], title='PACF |returns|²')
plt.tight_layout()
plt.show()

# Типичная картина:
# - ACF доходностей ≈ 0 для всех лагов (почти нет предсказуемости)
# - ACF |returns|² значительно > 0 для многих лагов (волатильность кластеризуется!)
# Это т.н. "stylized fact" финансовых рынков
```

**Стилизованные факты финансовых временных рядов:**
1. Доходности почти не автокоррелированы
2. Абсолютные/квадратичные доходности имеют сильную автокорреляцию (volatility clustering)
3. Толстые хвосты
4. Leverage effect: отрицательные шоки увеличивают волатильность сильнее, чем положительные
5. Агрегация ведёт к нормальности (недельные доходности ≈ нормальнее дневных)

### 3.3 Модели: расширенный арсенал

#### ARIMA и auto_arima

```python
from pmdarima import auto_arima

# auto_arima подбирает (p,d,q) автоматически по AIC/BIC
model = auto_arima(
    returns,
    start_p=0, max_p=5,
    start_q=0, max_q=5,
    d=None,  # автоматический выбор d
    seasonal=False,
    stepwise=True,
    trace=True,
    error_action='ignore',
    suppress_warnings=True,
    information_criterion='bic'  # BIC штрафует за сложность сильнее AIC
)
print(model.summary())
print(f"Порядок: {model.order}")  # обычно ARIMA(0,0,0) или (1,0,1) для доходностей
```

#### GARCH и его вариации

```python
from arch import arch_model

# GARCH(1,1) — базовая модель волатильности
# σ²_t = ω + α·ε²_{t-1} + β·σ²_{t-1}
garch = arch_model(returns * 100, vol='Garch', p=1, q=1, dist='t')
res_garch = garch.fit(disp='off')
print(res_garch.summary())

# EGARCH — учитывает leverage effect (асимметричную реакцию на шоки)
egarch = arch_model(returns * 100, vol='EGARCH', p=1, q=1, dist='t')
res_egarch = egarch.fit(disp='off')

# GJR-GARCH — другой способ моделировать асимметрию
gjr = arch_model(returns * 100, vol='Garch', p=1, o=1, q=1, dist='t')
res_gjr = gjr.fit(disp='off')

# Сравнение по BIC
for name, res in [('GARCH', res_garch), ('EGARCH', res_egarch), ('GJR', res_gjr)]:
    print(f"{name}: BIC={res.bic:.2f}")

# Прогноз волатильности на 10 дней
forecast = res_egarch.forecast(horizon=10, reindex=False)
vol_forecast = np.sqrt(forecast.variance.values[-1]) / 100  # обратно из %
print(f"Прогноз vol на 1-10 дней: {vol_forecast}")
```

#### HAR-RV (Heterogeneous Autoregressive Realized Volatility)

Простая, но мощная модель для прогнозирования волатильности. Использует realized volatility на разных горизонтах:

```python
def har_rv_model(rv_series):
    """
    HAR-RV: RV_t = c + β_d*RV_{t-1} + β_w*RV_{t-5:t-1} + β_m*RV_{t-22:t-1}
    rv_series: ряд реализованной волатильности (дневной)
    """
    df = pd.DataFrame({'rv': rv_series})
    df['rv_d'] = df['rv'].shift(1)  # вчерашняя
    df['rv_w'] = df['rv'].rolling(5).mean().shift(1)  # средняя за неделю
    df['rv_m'] = df['rv'].rolling(22).mean().shift(1)  # средняя за месяц
    df = df.dropna()

    X = sm.add_constant(df[['rv_d', 'rv_w', 'rv_m']])
    model = sm.OLS(df['rv'], X).fit(cov_type='HAC', cov_kwds={'maxlags': 5})
    print(model.summary())
    return model

# Realized volatility из 5-минутных данных
rv = intraday_returns.resample('D').apply(lambda x: np.sqrt(np.sum(x**2)))
har_model = har_rv_model(rv)
```

#### VAR (Vector Autoregression)

Моделирует несколько рядов одновременно. Каждый ряд зависит от прошлых значений *всех* рядов.

```python
from statsmodels.tsa.api import VAR

# Пример: доходности, объём, спред
data = df[['returns', 'volume_change', 'spread_change']].dropna()

model = VAR(data)
# Выбираем порядок по информационному критерию
result = model.select_order(maxlags=10)
print(result.summary())

# Подбираем модель с оптимальным лагом
var_result = model.fit(result.selected_orders['bic'])
print(var_result.summary())

# Impulse Response Function — как шок одной переменной влияет на другие?
irf = var_result.irf(periods=20)
irf.plot()
plt.show()

# Forecast Error Variance Decomposition — какая доля вариации объяснена каждой переменной?
fevd = var_result.fevd(periods=20)
fevd.plot()
plt.show()
```

### 3.4 Коинтеграция: подробно

```python
from statsmodels.tsa.stattools import coint
from statsmodels.tsa.vector_ar.vecm import coint_johansen, VECM

# Энгл-Грейнджер тест (для пары рядов)
score, p_value, _ = coint(prices_a, prices_b)
print(f"Коинтеграция p-value: {p_value:.4f}")

# Йохансен (для нескольких рядов)
result = coint_johansen(prices_df, det_order=0, k_ar_diff=1)
for i in range(len(result.lr1)):
    print(f"r ≤ {i}: trace stat={result.lr1[i]:.2f}, "
          f"critical 95%={result.cvt[i, 1]:.2f}, "
          f"{'reject' if result.lr1[i] > result.cvt[i, 1] else 'fail to reject'}")

# Вектор коинтеграции — коэффициенты для построения спреда
coint_vector = result.evec[:, 0]  # первый вектор
print(f"Коинтеграционный вектор: {coint_vector}")

# Построение спреда и торговый сигнал
spread = prices_df.values @ coint_vector
spread = pd.Series(spread, index=prices_df.index)

# Half-life of mean reversion
def half_life(spread):
    lag = spread.shift(1).dropna()
    delta = spread.diff().dropna()
    model = sm.OLS(delta, sm.add_constant(lag.iloc[1:])).fit()
    hl = -np.log(2) / model.params.iloc[1]
    return hl

hl = half_life(spread)
print(f"Half-life: {hl:.1f} дней")

# Z-score для торговых сигналов
lookback = int(hl)
z = (spread - spread.rolling(lookback).mean()) / spread.rolling(lookback).std()

# VECM — регрессия с коинтеграцией (если ряды коинтегрированы, VAR — неправильная модель)
vecm = VECM(prices_df, k_ar_diff=1, coint_rank=1).fit()
print(vecm.summary())
```

### 3.5 Спектральный анализ

Ищем периодические компоненты (циклы, сезонность) в частотной области:

```python
from scipy.signal import periodogram, welch

# Периодограмма — оценка спектральной плотности
freqs, psd = periodogram(returns, fs=252)  # fs=252 дня/год

# Welch — сглаженная версия (более устойчивая)
freqs_w, psd_w = welch(returns, fs=252, nperseg=256)

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.semilogy(freqs, psd)
plt.xlabel('Частота (циклов/год)')
plt.ylabel('Спектральная плотность')
plt.title('Периодограмма')

plt.subplot(1, 2, 2)
plt.semilogy(freqs_w, psd_w)
plt.xlabel('Частота (циклов/год)')
plt.title('Welch PSD')
plt.tight_layout()
plt.show()

# Поиск доминирующих частот
top_freq_idx = np.argsort(psd_w)[-5:]
for idx in top_freq_idx:
    period = 1 / freqs_w[idx] if freqs_w[idx] > 0 else np.inf
    print(f"Частота: {freqs_w[idx]:.2f} циклов/год, период: {period:.1f} дней")
```

### 3.6 Fractional Differentiation (дробное дифференцирование)

Из книги Marcos Lopez de Prado. Проблема: цены нестационарны (d=0), а доходности (d=1) теряют информацию о уровнях. Дробное d ∈ (0, 1) — компромисс: делает ряд стационарным, сохраняя часть "памяти".

```python
def frac_diff(series, d, threshold=1e-5):
    """
    Дробное дифференцирование порядка d.
    threshold — обрезка весов (для скорости).
    """
    # Веса: w_k = -w_{k-1} * (d - k + 1) / k
    weights = [1.0]
    k = 1
    while True:
        w = -weights[-1] * (d - k + 1) / k
        if abs(w) < threshold:
            break
        weights.append(w)
        k += 1
    weights = np.array(weights[::-1])

    # Свёртка
    result = np.convolve(series.values, weights, mode='valid')
    return pd.Series(result, index=series.index[len(weights)-1:])

# Находим минимальное d, при котором ряд становится стационарен
for d in np.arange(0, 1.1, 0.1):
    fd = frac_diff(prices, d)
    adf_p = adfuller(fd)[1]
    corr_with_orig = fd.corr(prices.reindex(fd.index))
    print(f"d={d:.1f}: ADF p={adf_p:.4f}, corr с ценой={corr_with_orig:.3f}")
    # Ищем наименьшее d с p < 0.05

# Типично d ≈ 0.3-0.5 — ряд стационарен, но сохраняет 80-90% корреляции с ценой
```

### 3.7 Wavelet анализ

Вейвлеты дают разложение по времени И частоте одновременно (спектральный анализ даёт только частоту):

```python
import pywt

# Continuous Wavelet Transform
scales = np.arange(1, 128)
coeff, freqs = pywt.cwt(returns.values, scales, 'morl')

plt.imshow(np.abs(coeff), aspect='auto', cmap='jet',
           extent=[0, len(returns), freqs[-1], freqs[0]])
plt.colorbar(label='Magnitude')
plt.ylabel('Частота')
plt.xlabel('Время')
plt.title('Wavelet Scalogram')
plt.show()

# Discrete Wavelet Transform — для декомпозиции на уровни
coeffs = pywt.wavedec(returns.values, 'db4', level=5)
# coeffs[0] — приближение (тренд)
# coeffs[1:] — детали (от низких частот к высоким)

# Можно использовать для denoising:
# Обнуляем мелкие коэффициенты (шум) и восстанавливаем сигнал
threshold = np.std(coeffs[-1]) * np.sqrt(2 * np.log(len(returns)))
coeffs_denoised = [pywt.threshold(c, threshold, mode='soft') for c in coeffs]
denoised = pywt.waverec(coeffs_denoised, 'db4')
```

---

## 4. Стохастические процессы и стохастическое исчисление

### 4.1 Броуновское движение (Wiener process)

Формальное определение:
- W(0) = 0
- W(t) - W(s) ~ N(0, t-s) для t > s
- Приращения независимы
- Пути непрерывны (но нигде не дифференцируемы!)

```python
def simulate_brownian(T, N, n_paths=5):
    """Симулирует n_paths траекторий броуновского движения."""
    dt = T / N
    t = np.linspace(0, T, N + 1)
    dW = np.random.normal(0, np.sqrt(dt), (n_paths, N))
    W = np.zeros((n_paths, N + 1))
    W[:, 1:] = np.cumsum(dW, axis=1)

    plt.figure(figsize=(12, 5))
    for i in range(n_paths):
        plt.plot(t, W[i], alpha=0.7)
    plt.title('Broунoвское движение (5 траекторий)')
    plt.xlabel('Время')
    plt.show()

    return t, W

t, W = simulate_brownian(1, 1000, 5)
```

### 4.2 Стохастическое исчисление Ito

#### Зачем это нужно?

Обычное исчисление (Ньютона-Лейбница) не работает для стохастических процессов, потому что броуновское движение нигде не дифференцируемо, а его квадратичная вариация ненулевая.

#### Лемма Ito — "chain rule" для стохастических процессов

Если X_t — процесс Ito (dX = μdt + σdW) и f(t, x) — гладкая функция, то:

```
df = (∂f/∂t + μ·∂f/∂x + ½σ²·∂²f/∂x²)dt + σ·∂f/∂x·dW
```

Дополнительный член **½σ²·∂²f/∂x²** — это то, чего нет в обычном исчислении. Именно он даёт volatility drag.

**Пример: вывод GBM**

Если S_t — цена актива и dS/S = μdt + σdW, то что происходит с log(S)?

Применяя Ito с f(S) = log(S):
```
d(log S) = (μ - σ²/2)dt + σdW
```

Минус σ²/2 — это volatility drag. Средняя *геометрическая* доходность всегда ниже *арифметической* на σ²/2.

```python
# Демонстрация леммы Ito численно
def verify_ito_lemma(S0=100, mu=0.1, sigma=0.3, T=1, N=10000):
    """Проверяем формулу Ito для log(S)."""
    dt = T / N
    S = np.zeros(N + 1)
    S[0] = S0

    # Симулируем GBM
    for i in range(N):
        dW = np.random.normal(0, np.sqrt(dt))
        S[i+1] = S[i] * (1 + mu*dt + sigma*dW)

    # log(S_T) - log(S_0) = ∫(μ - σ²/2)dt + ∫σdW
    log_return_actual = np.log(S[-1] / S[0])
    log_return_expected = (mu - sigma**2/2) * T  # + случайная часть

    print(f"Фактический log-return: {log_return_actual:.4f}")
    print(f"E[log-return] по Ito: {log_return_expected:.4f}")
    print(f"Volatility drag: {sigma**2/2:.4f}")

# Запускаем 10000 раз и смотрим среднее
log_returns_sim = []
for _ in range(10000):
    dt = 1/252
    dW = np.random.normal(0, np.sqrt(dt), 252)
    log_return = sum((0.1 - 0.3**2/2)*dt + 0.3*dw for dw in dW)
    log_returns_sim.append(log_return)

print(f"Средний log-return (10K симуляций): {np.mean(log_returns_sim):.4f}")
print(f"Теория (μ - σ²/2): {0.1 - 0.3**2/2:.4f}")
```

### 4.3 Модели стохастической волатильности

GBM предполагает постоянную волатильность. В реальности волатильность — сама случайный процесс.

#### Модель Хестона (Heston)

```
dS = μ·S·dt + √v·S·dW_1
dv = κ(θ - v)dt + ξ√v·dW_2
corr(dW_1, dW_2) = ρ
```

- v — мгновенная дисперсия (случайный процесс)
- κ — скорость возврата волатильности к среднему
- θ — долгосрочная средняя дисперсия
- ξ — "волатильность волатильности" (vol of vol)
- ρ — корреляция между ценой и волатильностью (обычно ρ < 0 — leverage effect)

```python
def simulate_heston(S0, v0, mu, kappa, theta, xi, rho, T, N, n_paths=1000):
    """Симулирует модель Хестона."""
    dt = T / N
    S = np.zeros((n_paths, N + 1))
    v = np.zeros((n_paths, N + 1))
    S[:, 0] = S0
    v[:, 0] = v0

    for t in range(N):
        # Коррелированные brownian motions
        z1 = np.random.standard_normal(n_paths)
        z2 = rho * z1 + np.sqrt(1 - rho**2) * np.random.standard_normal(n_paths)

        v_pos = np.maximum(v[:, t], 0)  # Truncation scheme
        S[:, t+1] = S[:, t] * np.exp(
            (mu - 0.5*v_pos)*dt + np.sqrt(v_pos*dt)*z1
        )
        v[:, t+1] = v[:, t] + kappa*(theta - v_pos)*dt + xi*np.sqrt(v_pos*dt)*z2

    return S, v

# Типичные параметры для S&P 500
S, v = simulate_heston(
    S0=100, v0=0.04,  # начальная vol = 20%
    mu=0.05,
    kappa=2.0,   # скорость возврата
    theta=0.04,  # долгосрочная дисперсия (vol = 20%)
    xi=0.3,      # vol of vol
    rho=-0.7,    # leverage effect
    T=1, N=252
)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
for i in range(min(10, len(S))):
    ax1.plot(S[i], alpha=0.5)
    ax2.plot(np.sqrt(v[i]) * 100, alpha=0.5)
ax1.set_title('Цена (Heston)')
ax2.set_title('Волатильность % (Heston)')
ax2.set_xlabel('Дни')
plt.tight_layout()
plt.show()
```

### 4.4 Процессы с прыжками (Jump-Diffusion)

#### Модель Мертона

Добавляет к GBM пуассоновские прыжки (earnings, flash crash, черные лебеди):

```python
def simulate_merton(S0, mu, sigma, lam, jump_mu, jump_sigma, T, N, n_paths=1000):
    """
    Merton Jump Diffusion.
    lam: интенсивность прыжков (среднее кол-во в год)
    jump_mu, jump_sigma: параметры размера прыжков (лог-нормальные)
    """
    dt = T / N
    S = np.zeros((n_paths, N + 1))
    S[:, 0] = S0

    for t in range(N):
        # Диффузная часть
        z = np.random.standard_normal(n_paths)
        diffusion = (mu - 0.5*sigma**2)*dt + sigma*np.sqrt(dt)*z

        # Прыжки (пуассоновские)
        n_jumps = np.random.poisson(lam * dt, n_paths)
        jump_sizes = np.array([
            np.sum(np.random.normal(jump_mu, jump_sigma, max(nj, 0)))
            if nj > 0 else 0
            for nj in n_jumps
        ])

        S[:, t+1] = S[:, t] * np.exp(diffusion + jump_sizes)

    return S

# Параметры: 2 прыжка в год, средний размер -3%, std 5%
S_jump = simulate_merton(100, 0.1, 0.15, 2, -0.03, 0.05, 1, 252)
```

### 4.5 Процесс Орнштейна-Уленбека и mean-reversion

```python
def simulate_ou(x0, theta, mu, sigma, T, N, n_paths=1000):
    """
    Ornstein-Uhlenbeck: dX = θ(μ - X)dt + σdW
    Основа pairs trading и статистического арбитража.
    """
    dt = T / N
    X = np.zeros((n_paths, N + 1))
    X[:, 0] = x0

    for t in range(N):
        dW = np.random.standard_normal(n_paths) * np.sqrt(dt)
        X[:, t+1] = X[:, t] + theta * (mu - X[:, t]) * dt + sigma * dW

    return X

# Оценка параметров OU из данных
def fit_ou(series, dt=1/252):
    """Оценивает параметры OU процесса из наблюдений (MLE)."""
    x = series.values
    n = len(x)

    # AR(1): X_t = a + b*X_{t-1} + eps
    x_lag = x[:-1]
    x_lead = x[1:]

    b = np.sum((x_lag - x_lag.mean()) * (x_lead - x_lead.mean())) / np.sum((x_lag - x_lag.mean())**2)
    a = x_lead.mean() - b * x_lag.mean()
    residuals = x_lead - a - b * x_lag

    # Перевод AR(1) → OU
    theta = -np.log(b) / dt
    mu = a / (1 - b)
    sigma = np.std(residuals) * np.sqrt(-2 * np.log(b) / (dt * (1 - b**2)))

    half_life = np.log(2) / theta

    print(f"θ (скорость возврата): {theta:.4f}")
    print(f"μ (долгосрочное среднее): {mu:.4f}")
    print(f"σ: {sigma:.4f}")
    print(f"Half-life: {half_life:.1f} дней")
    print(f"Стационарная дисперсия: {sigma**2/(2*theta):.6f}")

    return theta, mu, sigma, half_life

params = fit_ou(spread_series)
```

### 4.6 Lévy процессы

Обобщение броуновского движения + пуассоновского процесса. Любой процесс с независимыми стационарными приращениями — Lévy процесс. Характеризуется **триплетом Lévy** (b, σ², ν), где ν — мера прыжков (Lévy measure).

**Зачем:** более реалистичное моделирование, чем GBM. Включает и диффузию, и прыжки, и бесконечную активность.

```python
# Variance Gamma Process — популярный Lévy процесс для финансов
# = Brownian motion, подчинённый гамма-процессу (случайное время)
def simulate_vg(S0, mu, sigma, nu, T, N, n_paths=1000):
    """
    Variance Gamma Process.
    nu: параметр формы (дисперсия гамма-времени)
    Большее nu → толстые хвосты
    """
    dt = T / N
    S = np.zeros((n_paths, N + 1))
    S[:, 0] = S0

    for t in range(N):
        # Гамма-время
        gamma_t = np.random.gamma(shape=dt/nu, scale=nu, size=n_paths)
        # Broунoвское движение со случайным временем
        dX = mu * gamma_t + sigma * np.sqrt(gamma_t) * np.random.standard_normal(n_paths)
        S[:, t+1] = S[:, t] * np.exp(dX)

    return S
```

### 4.7 Параметр Хёрста: подробнее

```python
# Несколько методов оценки H:

# 1. R/S анализ (классический)
def hurst_rs(series):
    n = len(series)
    lags = range(20, min(n//2, 500))
    rs_values = []
    for lag in lags:
        rs_list = []
        for start in range(0, n - lag, lag):
            sub = series[start:start+lag]
            mean = sub.mean()
            cumdev = np.cumsum(sub - mean)
            R = cumdev.max() - cumdev.min()
            S = sub.std()
            if S > 0:
                rs_list.append(R / S)
        if rs_list:
            rs_values.append((lag, np.mean(rs_list)))
    log_lags = np.log([x[0] for x in rs_values])
    log_rs = np.log([x[1] for x in rs_values])
    H = np.polyfit(log_lags, log_rs, 1)[0]
    return H

# 2. DFA (Detrended Fluctuation Analysis) — более устойчив
def hurst_dfa(series):
    """Detrended Fluctuation Analysis."""
    y = np.cumsum(series - np.mean(series))
    n = len(y)

    scales = np.unique(np.logspace(1, np.log10(n//4), 20).astype(int))
    fluctuations = []

    for scale in scales:
        n_segments = n // scale
        rms = []
        for i in range(n_segments):
            segment = y[i*scale:(i+1)*scale]
            x = np.arange(scale)
            # Линейный тренд (можно квадратичный для DFA-2)
            coeffs = np.polyfit(x, segment, 1)
            trend = np.polyval(coeffs, x)
            rms.append(np.sqrt(np.mean((segment - trend)**2)))
        fluctuations.append(np.mean(rms))

    log_scales = np.log(scales)
    log_fluct = np.log(fluctuations)
    H = np.polyfit(log_scales, log_fluct, 1)[0]
    return H

H_rs = hurst_rs(returns.values)
H_dfa = hurst_dfa(returns.values)
print(f"Hurst (R/S): {H_rs:.3f}")
print(f"Hurst (DFA): {H_dfa:.3f}")

# Интерпретация и стратегии:
# H ≈ 0.5: random walk → нет edge
# H > 0.5: persistence → trend-following
# H < 0.5: anti-persistence → mean-reversion
# H может меняться со временем → скользящий Hurst

rolling_hurst = pd.Series(
    [hurst_dfa(returns.values[max(0,i-252):i])
     for i in range(252, len(returns))],
    index=returns.index[252:]
)
rolling_hurst.plot(title='Rolling Hurst exponent (252d)')
plt.axhline(0.5, color='red', linestyle='--')
plt.show()
```

---

## 5. Линейная алгебра и оптимизация

### 5.1 Зачем линейная алгебра трейдеру?

Портфель — это вектор весов. Доходности активов — вектор. Ковариации — матрица. Оптимизация портфеля — операция с матрицами.

### 5.2 Ковариационная матрица и её проблемы

```python
# Выборочная ковариационная матрица
cov_matrix = returns_df.cov() * 252  # Годовая

# Проблема 1: при N > T (активов больше, чем наблюдений) матрица вырождена
# Проблема 2: выборочная оценка зашумлена — мусорные собственные значения

# Решение 1: Shrinkage (сжатие Ledoit-Wolf)
from sklearn.covariance import LedoitWolf

lw = LedoitWolf().fit(returns_df)
cov_shrunk = lw.covariance_ * 252
print(f"Shrinkage intensity: {lw.shrinkage_:.4f}")
# Чем ближе к 1, тем сильнее сжатие к диагональной матрице

# Решение 2: Random Matrix Theory — очистка от шума
def marcenko_pastur_clean(returns_df, q=None):
    """
    Очистка ковариационной матрицы по Marchenko-Pastur.
    Собственные значения ниже порога MP — это шум.
    """
    T, N = returns_df.shape
    if q is None:
        q = T / N

    # Границы Marchenko-Pastur
    lambda_plus = (1 + 1/np.sqrt(q))**2
    lambda_minus = (1 - 1/np.sqrt(q))**2

    corr = returns_df.corr()
    eigenvalues, eigenvectors = np.linalg.eigh(corr)

    # Заменяем шумовые eigenvalues средним
    noise_mask = eigenvalues < lambda_plus
    signal_eigenvalues = eigenvalues.copy()
    signal_eigenvalues[noise_mask] = np.mean(eigenvalues[noise_mask])

    # Восстанавливаем матрицу
    cleaned_corr = eigenvectors @ np.diag(signal_eigenvalues) @ eigenvectors.T
    # Нормализуем диагональ
    d = np.sqrt(np.diag(cleaned_corr))
    cleaned_corr = cleaned_corr / np.outer(d, d)

    n_signal = (~noise_mask).sum()
    print(f"Сигнальных компонент: {n_signal} из {N}")
    return cleaned_corr

cleaned_corr = marcenko_pastur_clean(returns_df)
```

### 5.3 PCA (Principal Component Analysis)

Уменьшение размерности. Первые компоненты обычно = рыночные факторы.

```python
from sklearn.decomposition import PCA

pca = PCA(n_components=10)
pca.fit(returns_df)

print("Объяснённая дисперсия:")
for i, var in enumerate(pca.explained_variance_ratio_[:10]):
    print(f"  PC{i+1}: {var:.2%} (cumulative: {sum(pca.explained_variance_ratio_[:i+1]):.2%})")

# Типично для акций:
# PC1 ≈ 40-60% — рыночный фактор (все растут/падают вместе)
# PC2 ≈ 5-15% — sector rotation
# PC3+ ≈ 2-5% каждый
# Хвост — шум

# Факторные нагрузки первой компоненты
loadings = pd.Series(pca.components_[0], index=returns_df.columns)
loadings.sort_values().plot(kind='barh', title='PC1 loadings (≈ market factor)')
plt.show()

# Statistical arbitrage: торгуем residuals после вычитания первых k компонент
k = 3  # убираем первые 3 фактора
factor_returns = pca.transform(returns_df)[:, :k]
residuals = returns_df - pd.DataFrame(
    pca.inverse_transform(
        np.hstack([factor_returns, np.zeros((len(returns_df), returns_df.shape[1] - k))])
    ),
    index=returns_df.index, columns=returns_df.columns
)
# residuals — идиосинкратические доходности, потенциально mean-reverting
```

### 5.4 Оптимизация портфеля

#### Markowitz (Mean-Variance Optimization)

```python
from scipy.optimize import minimize

def markowitz_optimize(returns_df, target_return=None, risk_free_rate=0.02):
    """
    Оптимизация портфеля по Марковицу.
    """
    mu = returns_df.mean() * 252
    cov = LedoitWolf().fit(returns_df).covariance_ * 252
    n = len(mu)

    def neg_sharpe(weights):
        port_return = weights @ mu
        port_vol = np.sqrt(weights @ cov @ weights)
        return -(port_return - risk_free_rate) / port_vol

    # Ограничения: веса суммируются в 1
    constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
    # Границы: 0 ≤ w ≤ 0.20 (max 20% на один актив)
    bounds = [(0, 0.20)] * n

    result = minimize(neg_sharpe, np.ones(n)/n,
                     method='SLSQP', bounds=bounds, constraints=constraints)

    weights = pd.Series(result.x, index=returns_df.columns)
    port_return = weights @ mu
    port_vol = np.sqrt(weights @ cov @ weights)
    sharpe = (port_return - risk_free_rate) / port_vol

    print(f"Return: {port_return:.2%}")
    print(f"Volatility: {port_vol:.2%}")
    print(f"Sharpe: {sharpe:.2f}")
    print(f"Weights:\n{weights[weights > 0.001].sort_values(ascending=False)}")

    return weights

weights = markowitz_optimize(returns_df)
```

**Проблемы Марковица:**
1. Очень чувствителен к входным данным (особенно expected returns)
2. Часто даёт экстремальные веса
3. Средние доходности оценены с огромной ошибкой

#### Black-Litterman — решение проблемы входных данных

Байесовский подход: начинаем с рыночного равновесия (implied returns), затем добавляем свои "views":

```python
def black_litterman(cov, market_caps, risk_aversion=2.5,
                    views_matrix=None, views_returns=None, views_confidence=None):
    """
    Black-Litterman model.
    market_caps: рыночные капитализации (определяют начальные веса)
    views_matrix: P матрица (какие активы в каждом view)
    views_returns: Q вектор (ожидаемые returns по каждому view)
    views_confidence: Ω матрица (неуверенность в views)
    """
    # Рыночные веса
    w_mkt = market_caps / market_caps.sum()

    # Implied equilibrium returns
    tau = 0.05  # скалирующий фактор неопределённости
    pi = risk_aversion * cov @ w_mkt

    if views_matrix is not None:
        P = views_matrix
        Q = views_returns
        Omega = views_confidence if views_confidence is not None else np.diag(np.diag(tau * P @ cov @ P.T))

        # Posterior
        M = np.linalg.inv(np.linalg.inv(tau * cov) + P.T @ np.linalg.inv(Omega) @ P)
        mu_bl = M @ (np.linalg.inv(tau * cov) @ pi + P.T @ np.linalg.inv(Omega) @ Q)
    else:
        mu_bl = pi

    # Оптимальные веса
    w_bl = np.linalg.inv(risk_aversion * cov) @ mu_bl
    return w_bl, mu_bl
```

#### Risk Parity

Каждый актив вносит одинаковый вклад в общий риск портфеля:

```python
def risk_parity(cov):
    """Risk Parity — равный вклад в риск."""
    n = cov.shape[0]

    def risk_budget_objective(weights):
        port_vol = np.sqrt(weights @ cov @ weights)
        marginal_risk = cov @ weights / port_vol
        risk_contribution = weights * marginal_risk
        # Хотим, чтобы все risk contributions были равны
        target = port_vol / n
        return np.sum((risk_contribution - target)**2)

    constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
    bounds = [(0.01, None)] * n

    result = minimize(risk_budget_objective, np.ones(n)/n,
                     method='SLSQP', bounds=bounds, constraints=constraints)
    return result.x

# Risk Parity часто работает лучше Марковица на практике,
# потому что не зависит от оценок expected returns (только от cov)
```

### 5.5 Convex optimization tools

```python
import cvxpy as cp

# CVXPY — декларативная оптимизация. Описываешь задачу, решатель находит решение.
def portfolio_optimize_cvxpy(mu, cov, gamma=1.0, max_weight=0.1):
    """
    Оптимизация портфеля через CVXPY.
    Гибкая формулировка — легко добавлять ограничения.
    """
    n = len(mu)
    w = cp.Variable(n)

    # Целевая функция: maximize (return - gamma * risk)
    ret = mu @ w
    risk = cp.quad_form(w, cov)
    objective = cp.Maximize(ret - gamma * risk)

    # Ограничения
    constraints = [
        cp.sum(w) == 1,           # полная инвестированность
        w >= 0,                    # без шортов
        w <= max_weight,           # max вес
        # Легко добавить:
        # cp.norm(w, 1) <= 1.5,   # leverage ≤ 1.5
        # w @ sector_mask <= 0.3, # max 30% в секторе
    ]

    problem = cp.Problem(objective, constraints)
    problem.solve()

    if problem.status == 'optimal':
        return pd.Series(w.value, index=mu.index)
    else:
        raise ValueError(f"Optimization failed: {problem.status}")
```

---

## 6. Теория игр и микроструктура рынка

### 6.1 Равновесие Нэша и рынки

Равновесие Нэша — состояние, где ни одному участнику невыгодно менять стратегию в одностороннем порядке.

**Для рынков это означает:**
- Если паттерн известен → его эксплуатация → паттерн исчезает → равновесие
- Гипотеза эффективного рынка — это по сути утверждение о равновесии Нэша
- Альфа существует в **неравновесных** состояниях — когда рынок ещё не адаптировался

**Три формы эффективности (Fama):**
1. **Слабая**: цены отражают всю историческую информацию (технический анализ не работает)
2. **Средняя**: цены отражают всю публичную информацию (фундаментальный анализ не работает)
3. **Сильная**: цены отражают ВСЮ информацию, включая инсайдерскую (ничего не работает)

Реальность: рынки ≈ слабо-средне эффективны, но с **фрикциями** (transaction costs, информационные задержки, ограничения на short selling), которые создают возможности.

### 6.2 Аукционы и Order Book

```python
# Моделирование order book dynamics
class SimpleOrderBook:
    """
    Упрощённая модель стакана.
    Для реальной работы: LOBster data, ABIDES simulator
    """
    def __init__(self):
        self.bids = {}  # price → quantity
        self.asks = {}  # price → quantity

    def mid_price(self):
        if self.bids and self.asks:
            return (max(self.bids) + min(self.asks)) / 2
        return None

    def spread(self):
        if self.bids and self.asks:
            return min(self.asks) - max(self.bids)
        return None

    def book_imbalance(self, levels=5):
        """
        Дисбаланс стакана — предсказатель краткосрочного движения цены.
        imbalance > 0 → давление покупателей → цена скорее вырастет
        """
        bid_prices = sorted(self.bids.keys(), reverse=True)[:levels]
        ask_prices = sorted(self.asks.keys())[:levels]

        bid_vol = sum(self.bids[p] for p in bid_prices)
        ask_vol = sum(self.asks[p] for p in ask_prices)

        return (bid_vol - ask_vol) / (bid_vol + ask_vol)
```

### 6.3 Market Impact Models

```python
# Модель квадратного корня (эмпирическая)
def sqrt_impact(volume, adv, daily_vol, spread):
    """
    Market impact ≈ σ_daily × √(V/ADV)
    volume: объём ордера
    adv: average daily volume
    daily_vol: дневная волатильность
    spread: bid-ask spread
    """
    participation = volume / adv
    temporary_impact = daily_vol * np.sqrt(participation)
    permanent_impact = temporary_impact * 0.5  # ~50% permanent
    return {
        'temporary': temporary_impact,
        'permanent': permanent_impact,
        'spread_cost': spread / 2,
        'total': temporary_impact + spread / 2
    }

# Пример: покупаем 50K акций, ADV = 1M, vol = 2%, spread = 0.05%
impact = sqrt_impact(50000, 1000000, 0.02, 0.0005)
for k, v in impact.items():
    print(f"{k}: {v:.4f} ({v*100:.2f}%)")

# Kyle's Lambda — мера ценового воздействия
# ΔP = λ × order_flow, где λ — мера ликвидности
# Высокий λ = неликвидный рынок = большой impact
```

### 6.4 Optimal Execution: Almgren-Chriss

```python
def almgren_chriss(X0, T, n_steps, sigma, eta, gamma, alpha=0):
    """
    Оптимальное расписание исполнения Almgren-Chriss.
    X0: общий объём
    T: время исполнения
    sigma: волатильность
    eta: параметр временного impact
    gamma: неприятие риска
    alpha: drift (ожидаемое изменение цены)
    """
    tau = T / n_steps
    kappa = np.sqrt(gamma * sigma**2 / eta)

    # Оптимальная траектория
    trajectory = np.zeros(n_steps + 1)
    trajectory[0] = X0
    for j in range(1, n_steps + 1):
        t_j = j * tau
        trajectory[j] = X0 * np.sinh(kappa * (T - t_j)) / np.sinh(kappa * T)

    # Скорость торговли
    trade_rate = -np.diff(trajectory) / tau

    # Ожидаемая стоимость и дисперсия
    expected_cost = eta * np.sum(trade_rate**2) * tau
    variance = sigma**2 * np.sum(trajectory[:-1]**2 * tau)

    print(f"Expected cost: {expected_cost:.6f}")
    print(f"Risk (std): {np.sqrt(variance):.6f}")
    print(f"Trajectory: {trajectory.round(0)}")
    print(f"Trade rate: {trade_rate.round(0)}")

    return trajectory, trade_rate

# Пример: продаём 100K акций за 5 дней
traj, rate = almgren_chriss(
    X0=100000, T=5, n_steps=10,
    sigma=0.02, eta=0.0001, gamma=0.001
)
```

### 6.5 Adverse Selection и Maker-Taker Model

**Модель Glosten-Milgrom:** маркет-мейкер устанавливает спред, чтобы компенсировать убытки от торговли с информированными трейдерами.

```python
# Decomposition of spread: ROLL model
def roll_model(returns):
    """
    Оценка эффективного спреда по модели Ролла.
    Спред ≈ 2 × sqrt(-cov(r_t, r_{t-1}))
    """
    cov = np.cov(returns[1:], returns[:-1])[0, 1]
    if cov < 0:
        spread = 2 * np.sqrt(-cov)
    else:
        spread = 0  # модель не применима
    return spread

# PIN (Probability of Informed Trading) — мера информированной торговли
# Высокий PIN → больше informed traders → шире спреды
```

### 6.6 Alpha Decay и Capacity

**Alpha decay формула (упрощённо):**

```
Alpha(t) = Alpha(0) × exp(-λ × capacity_used / capacity_total)
```

Чем больше капитала эксплуатирует стратегию, тем меньше альфа.

```python
def estimate_strategy_capacity(avg_alpha, market_impact_coeff, avg_daily_volume):
    """
    Грубая оценка capacity стратегии.
    Точка, где market impact съедает всю альфу.
    """
    # alpha ≈ impact → alpha = impact_coeff * sqrt(V_strategy / V_market)
    # V_strategy = (alpha / impact_coeff)² × V_market
    capacity_shares = (avg_alpha / market_impact_coeff)**2 * avg_daily_volume
    return capacity_shares

# Пример: альфа = 0.1% на сделку, impact_coeff = 0.5%, ADV = 1M
cap = estimate_strategy_capacity(0.001, 0.005, 1000000)
print(f"Capacity: {cap:.0f} акций/день → ${cap * 50:,.0f}")
```

---

## 7. Риск-менеджмент

### 7.1 Когерентные меры риска

Мера риска ρ когерентна, если:
1. **Монотонность:** X ≤ Y ⇒ ρ(X) ≥ ρ(Y) (меньше доходность — больше риск)
2. **Субаддитивность:** ρ(X+Y) ≤ ρ(X) + ρ(Y) (диверсификация не увеличивает риск)
3. **Положительная однородность:** ρ(λX) = λρ(X)
4. **Трансляционная инвариантность:** ρ(X+c) = ρ(X) - c

**VaR — НЕ когерентен** (нарушает субаддитивность). Два актива с VaR=10 могут дать портфель с VaR=25.

**CVaR (Expected Shortfall) — когерентен.** Предпочтительная мера риска.

```python
def comprehensive_risk_metrics(returns, confidence=0.95):
    """Полный набор метрик риска."""
    alpha = 1 - confidence

    # VaR
    var_hist = np.percentile(returns, alpha * 100)
    var_param = stats.norm.ppf(alpha, returns.mean(), returns.std())
    var_cf = cornish_fisher_var(returns, alpha)  # с поправкой на skew/kurtosis

    # CVaR
    cvar = returns[returns <= var_hist].mean()

    # Drawdown metrics
    equity = (1 + returns).cumprod()
    peak = equity.expanding().max()
    drawdown = (equity - peak) / peak
    max_dd = drawdown.min()
    avg_dd = drawdown[drawdown < 0].mean()

    # Calmar, Sterling
    annual_return = returns.mean() * 252
    calmar = annual_return / abs(max_dd)
    sterling = annual_return / abs(avg_dd) if avg_dd != 0 else np.inf

    # Ulcer Index — RMS drawdown
    ulcer = np.sqrt(np.mean(drawdown**2))

    return {
        'VaR (hist)': var_hist,
        'VaR (param)': var_param,
        'VaR (CF)': var_cf,
        'CVaR': cvar,
        'Max Drawdown': max_dd,
        'Avg Drawdown': avg_dd,
        'Max DD Duration': drawdown_duration(equity),
        'Calmar': calmar,
        'Sterling': sterling,
        'Ulcer Index': ulcer,
    }

def cornish_fisher_var(returns, alpha=0.05):
    """VaR с поправкой Корниша-Фишера (учитывает skew и kurtosis)."""
    z = stats.norm.ppf(alpha)
    s = stats.skew(returns)
    k = stats.kurtosis(returns)
    z_cf = (z +
            (z**2 - 1) * s / 6 +
            (z**3 - 3*z) * k / 24 -
            (2*z**3 - 5*z) * s**2 / 36)
    return returns.mean() + z_cf * returns.std()
```

### 7.2 Критерий Келли: глубже

```python
def kelly_multiple_assets(mu, cov, risk_free=0):
    """
    Критерий Келли для портфеля активов.
    Оптимальные веса = Σ^(-1) × (μ - r_f)
    """
    excess = mu - risk_free / 252
    weights = np.linalg.inv(cov) @ excess
    return weights

# На практике Kelly даёт ОЧЕНЬ агрессивные веса
kelly_w = kelly_multiple_assets(returns_df.mean(), returns_df.cov())
print(f"Kelly weights (сумма абс. весов): {np.abs(kelly_w).sum():.1f}")
# Может быть 5-10x leverage!

# Fractional Kelly — практический выбор
for fraction in [1.0, 0.5, 0.25]:
    w = kelly_w * fraction
    port_return = w @ returns_df.mean() * 252
    port_vol = np.sqrt(w @ returns_df.cov().values @ w) * np.sqrt(252)
    print(f"f={fraction}: E[r]={port_return:.2%}, vol={port_vol:.2%}, "
          f"Sharpe={port_return/port_vol:.2f}, leverage={np.abs(w).sum():.1f}x")
```

**Почему Half-Kelly:**
- Full Kelly максимизирует E[log(wealth)], но с огромными просадками
- Кривая рост/риск очень асимметрична: при 1.5×Kelly growth rate ПАДАЕТ, а volatility РАСТЁТ
- Half-Kelly даёт ~75% growth rate при ~50% волатильности
- Quarter-Kelly — консервативный вариант для реальной торговли

### 7.3 Стресс-тестирование

```python
def historical_stress_test(returns, portfolio_weights=None):
    """Стресс-тест на исторических кризисах."""
    crises = {
        'Black Monday (1987)': ('1987-10-14', '1987-10-30'),
        'LTCM (1998)': ('1998-08-01', '1998-10-15'),
        'Dot-com crash': ('2000-03-10', '2002-10-09'),
        '9/11': ('2001-09-10', '2001-09-21'),
        'GFC': ('2008-09-01', '2009-03-09'),
        'Flash Crash (2010)': ('2010-05-06', '2010-05-07'),
        'COVID crash': ('2020-02-19', '2020-03-23'),
        'Rate hike shock (2022)': ('2022-01-03', '2022-10-12'),
    }

    print(f"{'Crisis':<30} {'Duration':>10} {'Max DD':>10} {'Return':>10}")
    print('-' * 65)

    for name, (start, end) in crises.items():
        try:
            crisis_r = returns[start:end]
            if len(crisis_r) < 2:
                continue
            equity = (1 + crisis_r).cumprod()
            max_dd = ((equity / equity.expanding().max()) - 1).min()
            total_r = (1 + crisis_r).prod() - 1
            print(f"{name:<30} {len(crisis_r):>8}d {max_dd:>10.2%} {total_r:>10.2%}")
        except Exception:
            continue

# Monte-Carlo стресс-тест с копулами (tail scenarios)
def mc_stress_test(returns_df, n_scenarios=10000, tail_percentile=1):
    """
    Генерируем сценарии из хвостов распределения.
    Используем t-копулу для реалистичной tail dependence.
    """
    # Простой подход: bootstrap из worst days
    worst_days = returns_df[
        returns_df.sum(axis=1) < returns_df.sum(axis=1).quantile(tail_percentile/100)
    ]
    print(f"Worst {tail_percentile}% дней ({len(worst_days)} дней):")
    print(worst_days.describe())
    return worst_days
```

### 7.4 Risk Budgeting

```python
def risk_contribution(weights, cov):
    """Вклад каждого актива в общий риск портфеля."""
    port_vol = np.sqrt(weights @ cov @ weights)
    marginal_risk = cov @ weights / port_vol
    risk_contrib = weights * marginal_risk
    risk_contrib_pct = risk_contrib / port_vol
    return risk_contrib_pct

# Пример
w = np.array([0.6, 0.3, 0.1])
cov = returns_df.cov().values * 252
rc = risk_contribution(w, cov)
for name, wt, r in zip(returns_df.columns, w, rc):
    print(f"{name}: weight={wt:.0%}, risk contribution={r:.2%}")
# Часто 60% в акциях вносят 90% риска
```

---

## 8. Машинное обучение в трейдинге

### 8.1 Почему ML в финансах сложнее, чем в CV/NLP

| Аспект | CV/NLP | Финансы |
|--------|--------|---------|
| Signal-to-noise | Высокий | Очень низкий |
| Стационарность | Данные стабильны | Распределения меняются |
| Объём данных | Миллионы примеров | Тысячи (дневных) |
| Независимость | Фото независимы | Временная зависимость |
| Overfitting risk | Средний | Экстремальный |
| Cross-validation | Стандартная | Только time-series split |
| Competition | Нет adversary | Другие ML-модели торгуют против тебя |

### 8.2 Правильная валидация

```python
from sklearn.model_selection import TimeSeriesSplit

# НИКОГДА не используй обычный KFold для временных рядов!

# Walk-Forward Validation — золотой стандарт
class WalkForwardCV:
    """
    Expanding или sliding window cross-validation.
    Опционально: gap между train и test (embargo period).
    """
    def __init__(self, n_splits=5, train_size=None, test_size=None, gap=0):
        self.n_splits = n_splits
        self.train_size = train_size
        self.test_size = test_size
        self.gap = gap  # дней между train и test (для предотвращения leakage)

    def split(self, X):
        n = len(X)
        test_size = self.test_size or n // (self.n_splits + 1)

        for i in range(self.n_splits):
            test_end = n - (self.n_splits - i - 1) * test_size
            test_start = test_end - test_size
            train_end = test_start - self.gap

            if self.train_size:
                train_start = max(0, train_end - self.train_size)
            else:
                train_start = 0

            yield (list(range(train_start, train_end)),
                   list(range(test_start, test_end)))

# Purged Cross-Validation (Lopez de Prado)
# Добавляет embargo period, чтобы предотвратить leakage из-за overlapping labels
class PurgedKFold:
    """
    Purged + Embargo cross-validation.
    Необходимо, когда labels создаются из overlapping windows.
    """
    def __init__(self, n_splits=5, embargo_pct=0.01):
        self.n_splits = n_splits
        self.embargo_pct = embargo_pct

    def split(self, X, labels_start=None, labels_end=None):
        n = len(X)
        embargo = int(n * self.embargo_pct)
        indices = np.arange(n)
        fold_size = n // self.n_splits

        for i in range(self.n_splits):
            test_start = i * fold_size
            test_end = min((i + 1) * fold_size, n)

            # Purge: убираем training samples, чьи labels пересекаются с test period
            # Embargo: убираем samples сразу после test period
            train_mask = np.ones(n, dtype=bool)
            train_mask[test_start:min(test_end + embargo, n)] = False

            yield indices[train_mask].tolist(), indices[test_start:test_end].tolist()
```

### 8.3 Feature Importance: за пределами model.feature_importances_

```python
# Mean Decrease Impurity (MDI) — встроенный, но biased к continuous features
mdi = model.feature_importances_

# Mean Decrease Accuracy (MDA) — permutation importance, более честный
from sklearn.inspection import permutation_importance

perm_imp = permutation_importance(model, X_test, y_test,
                                   n_repeats=10, random_state=42)
mda = pd.Series(perm_imp.importances_mean, index=feature_names)

# Single Feature Importance (SFI) — importance каждой фичи по отдельности
def single_feature_importance(model_class, X, y, cv):
    """Тренируем модель на каждой фиче отдельно."""
    sfi = {}
    for col in X.columns:
        scores = []
        for train_idx, test_idx in cv.split(X):
            m = model_class().fit(X[[col]].iloc[train_idx], y.iloc[train_idx])
            score = m.score(X[[col]].iloc[test_idx], y.iloc[test_idx])
            scores.append(score)
        sfi[col] = np.mean(scores)
    return pd.Series(sfi).sort_values(ascending=False)

# SHAP — самый информативный метод
import shap
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)
shap.summary_plot(shap_values, X_test)
```

### 8.4 Meta-labeling (Lopez de Prado)

Вместо того чтобы предсказывать направление, предсказываем **размер ставки** для существующей стратегии:

```python
def meta_labeling(primary_signals, returns, threshold=0):
    """
    Шаг 1: Primary model генерирует сигналы (buy/sell)
    Шаг 2: Meta model решает, следовать ли сигналу и с каким размером

    primary_signals: 1 (buy) или -1 (sell)
    returns: реальные доходности
    """
    # Метка: был ли primary signal правильным?
    meta_labels = np.sign(primary_signals * returns)
    # 1 = сигнал был правильным, -1 = неправильным

    # Для непрерывного размера:
    meta_labels_continuous = primary_signals * returns
    # Положительные = правильный сигнал, величина = прибыль

    return meta_labels

# Обучаем meta-model (classifier: следовать/не следовать сигналу)
# Это превращает проблему "предскажи направление" в "предскажи качество сигнала"
# Часто значительно проще!
```

### 8.5 Triple Barrier Labeling

```python
def triple_barrier_labels(prices, events, pt_sl=(1, 1), min_return=0,
                          vertical_barrier_days=10):
    """
    Triple Barrier Method (Lopez de Prado).
    Каждая точка входа имеет 3 барьера:
    1. Profit take (верхний)
    2. Stop loss (нижний)
    3. Max holding period (вертикальный)

    Лейбл = какой барьер сработал первым.
    """
    labels = []
    for idx in events.index:
        price_at_entry = prices[idx]
        vol = prices[:idx].pct_change().std()  # realized vol для масштабирования

        pt = price_at_entry * (1 + pt_sl[0] * vol)
        sl = price_at_entry * (1 - pt_sl[1] * vol)
        vb = idx + pd.Timedelta(days=vertical_barrier_days)

        future = prices[idx:vb]

        # Какой барьер сработал первым?
        pt_hit = future[future >= pt].index.min() if (future >= pt).any() else pd.NaT
        sl_hit = future[future <= sl].index.min() if (future <= sl).any() else pd.NaT

        if pt_hit < sl_hit:
            labels.append(1)   # profit take
        elif sl_hit < pt_hit:
            labels.append(-1)  # stop loss
        else:
            # Vertical barrier — смотрим return
            final_return = future.iloc[-1] / price_at_entry - 1
            labels.append(np.sign(final_return))

    return pd.Series(labels, index=events.index)
```

### 8.6 Использование LLM в торговых стратегиях

```python
# 1. Sentiment Analysis из новостей
def llm_sentiment(news_text, client):
    """
    Используем Claude/GPT для оценки sentiment.
    Преимущество перед VADER/TextBlob: понимает контекст, сарказм, финансовый жаргон.
    """
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=100,
        messages=[{
            "role": "user",
            "content": f"""Rate the sentiment of this financial news for the mentioned stock.
Reply with ONLY a JSON: {{"sentiment": float(-1 to 1), "confidence": float(0 to 1), "ticker": "XXX"}}

News: {news_text}"""
        }]
    )
    return json.loads(response.content[0].text)

# 2. Earnings call analysis — извлечение structured data
def analyze_earnings_call(transcript, client):
    """Извлекаем ключевые метрики из earnings call."""
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=500,
        messages=[{
            "role": "user",
            "content": f"""Extract from this earnings call transcript:
1. Revenue growth (%)
2. EPS surprise (actual vs estimate)
3. Forward guidance tone (bearish/neutral/bullish)
4. Key risks mentioned
5. New products/initiatives

Respond in JSON. Transcript:
{transcript[:5000]}"""
        }]
    )
    return json.loads(response.content[0].text)

# 3. Генерация trading ideas
def generate_hypotheses(market_context, client):
    """Генерация гипотез для тестирования."""
    response = client.messages.create(
        model="claude-opus-4-20250514",
        max_tokens=1000,
        messages=[{
            "role": "user",
            "content": f"""Given this market context, generate 5 testable quantitative trading hypotheses.
For each: describe the signal, expected mechanism, how to test, and potential pitfalls.

Context: {market_context}"""
        }]
    )
    return response.content[0].text

# 4. Code review для look-ahead bias
# LLM отлично справляется с поиском subtle bugs в backtesting code:
# "Review this backtest code for look-ahead bias, survivorship bias, or other data leakage"
```

---

## 9. Информационная теория

### 9.1 Энтропия: мера неопределённости

Энтропия Шеннона для дискретной случайной величины:

```
H(X) = -Σ p(x) × log₂(p(x))
```

Для непрерывных — дифференциальная энтропия:

```
h(X) = -∫ f(x) × log(f(x)) dx
```

```python
from scipy.stats import entropy, differential_entropy

# Дискретная энтропия: дискретизируем доходности
def market_entropy(returns, n_bins=20):
    """Энтропия рыночных доходностей — мера неопределённости."""
    hist, _ = np.histogram(returns, bins=n_bins, density=True)
    hist = hist[hist > 0]
    h = entropy(hist)  # натуральные логарифмы

    # Нормализованная энтропия (0 = полная определённость, 1 = макс. неопределённость)
    h_max = np.log(n_bins)
    h_norm = h / h_max

    return h, h_norm

h, h_norm = market_entropy(returns)
print(f"Энтропия: {h:.3f}")
print(f"Нормализованная: {h_norm:.3f}")

# Скользящая энтропия — детектирует изменения предсказуемости
rolling_entropy = returns.rolling(60).apply(
    lambda x: market_entropy(x)[1], raw=True
)
rolling_entropy.plot(title='Rolling Normalized Entropy (60d)')
plt.show()
# Низкая энтропия → рынок более предсказуем (сильный тренд или кластеризация)
```

### 9.2 Mutual Information: нелинейная зависимость

```python
from sklearn.feature_selection import mutual_info_regression, mutual_info_classif
from sklearn.neighbors import KernelDensity

# MI для feature selection — альтернатива корреляции
mi = mutual_info_regression(X, y, discrete_features=False, n_neighbors=7)
mi_series = pd.Series(mi, index=feature_names).sort_values(ascending=False)
print("Взаимная информация фичей с target:")
print(mi_series.head(15))

# Сравнение MI vs |correlation|
corr = X.corrwith(y).abs()
comparison = pd.DataFrame({'MI': mi_series, '|Corr|': corr}).dropna()
print(f"\nФичи с высоким MI, но низкой корреляцией (нелинейные зависимости):")
nonlinear = comparison[(comparison['MI'] > comparison['MI'].median()) &
                        (comparison['|Corr|'] < comparison['|Corr|'].median())]
print(nonlinear)
```

### 9.3 Transfer Entropy: направленная информация

```python
def transfer_entropy_simple(source, target, lag=1, n_bins=10):
    """
    Упрощённый расчёт Transfer Entropy.
    TE(X→Y) = H(Y_t | Y_{t-1}) - H(Y_t | Y_{t-1}, X_{t-1})

    Если TE(X→Y) > TE(Y→X), то X "вызывает" Y (в смысле Грейнджера, но нелинейно).
    """
    # Дискретизируем
    source_d = pd.qcut(source, n_bins, labels=False, duplicates='drop')
    target_d = pd.qcut(target, n_bins, labels=False, duplicates='drop')

    # Создаём лагированные переменные
    df = pd.DataFrame({
        'y_t': target_d.values[lag:],
        'y_lag': target_d.values[:-lag],
        'x_lag': source_d.values[:-lag]
    }).dropna()

    # Считаем условные энтропии через подсчёт частот
    from collections import Counter

    def conditional_entropy(df, target_col, condition_cols):
        groups = df.groupby(condition_cols)
        h = 0
        for name, group in groups:
            p_cond = len(group) / len(df)
            counts = Counter(group[target_col])
            total = sum(counts.values())
            h_cond = -sum(c/total * np.log2(c/total) for c in counts.values() if c > 0)
            h += p_cond * h_cond
        return h

    h_y_given_ylag = conditional_entropy(df, 'y_t', ['y_lag'])
    h_y_given_ylag_xlag = conditional_entropy(df, 'y_t', ['y_lag', 'x_lag'])

    te = h_y_given_ylag - h_y_given_ylag_xlag
    return te

# Пример: SPY → QQQ? QQQ → SPY?
te_spy_to_qqq = transfer_entropy_simple(returns_spy, returns_qqq)
te_qqq_to_spy = transfer_entropy_simple(returns_qqq, returns_spy)
print(f"TE(SPY → QQQ): {te_spy_to_qqq:.4f}")
print(f"TE(QQQ → SPY): {te_qqq_to_spy:.4f}")
net_te = te_spy_to_qqq - te_qqq_to_spy
print(f"Net TE: {net_te:.4f} ({'SPY leads' if net_te > 0 else 'QQQ leads'})")
```

### 9.4 Кросс-энтропия и KL-дивергенция

**KL-дивергенция** D_KL(P||Q) — насколько распределение Q отличается от P:

```python
def kl_divergence(p, q, n_bins=50):
    """KL-дивергенция между двумя выборками."""
    # Общая сетка
    all_data = np.concatenate([p, q])
    bins = np.linspace(all_data.min(), all_data.max(), n_bins)

    p_hist, _ = np.histogram(p, bins=bins, density=True)
    q_hist, _ = np.histogram(q, bins=bins, density=True)

    # Добавляем eps для избежания log(0)
    eps = 1e-10
    p_hist = p_hist + eps
    q_hist = q_hist + eps

    # Нормализуем
    p_hist = p_hist / p_hist.sum()
    q_hist = q_hist / q_hist.sum()

    kl = np.sum(p_hist * np.log(p_hist / q_hist))
    return kl

# Применение: обнаружение смены режима
# Если KL(returns_recent || returns_historical) высокий — режим сменился
for window in range(60, len(returns) - 60, 60):
    recent = returns.values[window:window+60]
    historical = returns.values[:window]
    kl = kl_divergence(recent, historical)
    if kl > 0.5:  # порог
        print(f"Regime change detected at index {window}, KL={kl:.3f}")
```

---

## 10. Численные методы и Монте-Карло

### 10.1 Монте-Карло симуляции

```python
def monte_carlo_portfolio(returns_df, weights, n_simulations=10000,
                          n_days=252, method='bootstrap'):
    """
    Монте-Карло симуляция будущего портфеля.
    method: 'bootstrap', 'parametric', 'block_bootstrap'
    """
    portfolio_returns = (returns_df * weights).sum(axis=1)

    if method == 'bootstrap':
        # Случайная выборка с возвращением из исторических доходностей
        simulations = np.random.choice(portfolio_returns, size=(n_simulations, n_days))

    elif method == 'parametric':
        # Генерируем из параметрического распределения (t-distribution)
        params = stats.t.fit(portfolio_returns)
        simulations = stats.t.rvs(*params, size=(n_simulations, n_days))

    elif method == 'block_bootstrap':
        # Блоки для сохранения автокорреляции
        block_size = 20
        n_blocks = n_days // block_size + 1
        simulations = np.zeros((n_simulations, n_days))
        for i in range(n_simulations):
            blocks = []
            for _ in range(n_blocks):
                start = np.random.randint(0, len(portfolio_returns) - block_size)
                blocks.append(portfolio_returns.values[start:start+block_size])
            simulations[i] = np.concatenate(blocks)[:n_days]

    # Кумулятивные пути
    equity_paths = np.cumprod(1 + simulations, axis=1)

    # Статистики
    final_values = equity_paths[:, -1]
    print(f"=== Monte Carlo ({method}, {n_simulations} sims, {n_days} days) ===")
    print(f"Median final: {np.median(final_values):.4f}")
    print(f"Mean final: {np.mean(final_values):.4f}")
    print(f"5th percentile: {np.percentile(final_values, 5):.4f}")
    print(f"95th percentile: {np.percentile(final_values, 95):.4f}")
    print(f"P(loss): {(final_values < 1).mean():.2%}")
    print(f"P(>20% gain): {(final_values > 1.20).mean():.2%}")

    # Max drawdown distribution
    max_dds = []
    for path in equity_paths:
        peak = np.maximum.accumulate(path)
        dd = (path - peak) / peak
        max_dds.append(dd.min())

    print(f"Expected max DD: {np.mean(max_dds):.2%}")
    print(f"95th percentile DD: {np.percentile(max_dds, 5):.2%}")

    return equity_paths
```

### 10.2 Importance Sampling

Стандартный MC плохо оценивает редкие события. Importance sampling "перевешивает" вероятности, чтобы чаще семплировать из хвостов:

```python
def var_importance_sampling(returns, alpha=0.01, n_samples=100000):
    """
    Оценка VaR через importance sampling.
    Сдвигаем распределение к хвосту для лучшей оценки.
    """
    mu, sigma = returns.mean(), returns.std()

    # Обычный MC
    samples_mc = np.random.normal(mu, sigma, n_samples)
    var_mc = np.percentile(samples_mc, alpha * 100)

    # Importance sampling: семплируем из сдвинутого распределения
    shift = -3 * sigma  # сдвигаем к левому хвосту
    samples_is = np.random.normal(mu + shift, sigma, n_samples)

    # Likelihood ratio (коррекция весов)
    log_weights = (-(samples_is - mu)**2 / (2*sigma**2) +
                   (samples_is - mu - shift)**2 / (2*sigma**2))
    weights = np.exp(log_weights)
    weights /= weights.sum()

    # Weighted percentile
    sorted_idx = np.argsort(samples_is)
    cumweights = np.cumsum(weights[sorted_idx])
    var_is = samples_is[sorted_idx[np.searchsorted(cumweights, alpha)]]

    print(f"VaR {alpha:.1%} (MC, {n_samples} samples): {var_mc:.6f}")
    print(f"VaR {alpha:.1%} (IS, {n_samples} samples): {var_is:.6f}")
    print(f"IS variance reduction: ~{np.std(samples_mc < var_mc) / np.std(weights * (samples_is < var_is)):.0f}x")

    return var_mc, var_is
```

### 10.3 Finite Difference Methods

Для ценообразования опционов и PDE:

```python
def black_scholes_fd(S_max, K, T, r, sigma, N_s=100, N_t=1000, option='call'):
    """
    Решение Black-Scholes PDE методом конечных разностей (implicit).
    Для случаев, когда нет аналитического решения (American options, barriers).
    """
    ds = S_max / N_s
    dt = T / N_t
    S = np.linspace(0, S_max, N_s + 1)

    # Terminal condition (payoff at expiry)
    if option == 'call':
        V = np.maximum(S - K, 0)
    else:
        V = np.maximum(K - S, 0)

    # Implicit scheme: A × V_{t} = V_{t+1}
    # Tri-diagonal matrix
    j = np.arange(1, N_s)
    a = 0.5 * dt * (sigma**2 * j**2 - r * j)
    b = 1 + dt * (sigma**2 * j**2 + r)
    c = 0.5 * dt * (-sigma**2 * j**2 - r * j)  # Fixed sign

    # Time stepping (backward)
    for _ in range(N_t):
        # Solve tridiagonal system
        V[1:-1] = thomas_algorithm(a, b, c, V[1:-1])
        # Boundary conditions
        V[0] = 0 if option == 'call' else K * np.exp(-r * dt)
        V[-1] = S_max - K if option == 'call' else 0

    return S, V
```

### 10.4 Оптимизация: практические алгоритмы

```python
from scipy.optimize import (minimize, differential_evolution,
                             basinhopping, dual_annealing)

def optimize_strategy_params(objective_fn, bounds, method='all'):
    """
    Сравнение алгоритмов оптимизации для торговых стратегий.
    objective_fn: функция(params) → negative Sharpe (минимизируем)
    bounds: [(low, high), ...] для каждого параметра
    """
    results = {}

    # Gradient-based (быстро, но локальный минимум)
    x0 = [(b[0] + b[1])/2 for b in bounds]
    res = minimize(objective_fn, x0, method='Nelder-Mead', bounds=bounds)
    results['Nelder-Mead'] = {'params': res.x, 'sharpe': -res.fun}

    # Differential Evolution (глобальный, без градиентов)
    res = differential_evolution(objective_fn, bounds, maxiter=100,
                                  seed=42, workers=-1)
    results['DE'] = {'params': res.x, 'sharpe': -res.fun}

    # Dual Annealing (глобальный, хорош для невыпуклых задач)
    res = dual_annealing(objective_fn, bounds, maxiter=100, seed=42)
    results['DA'] = {'params': res.x, 'sharpe': -res.fun}

    for name, r in results.items():
        print(f"{name}: Sharpe={r['sharpe']:.3f}, params={r['params'].round(4)}")

    # ВАЖНО: оптимизация параметров стратегии = подгонка под данные
    # Всегда валидируйте на out-of-sample!
    return results
```

---

## 11. Инструменты и библиотеки

### 11.1 Python-экосистема

| Категория | Библиотека | Назначение | Уровень |
|-----------|-----------|------------|---------|
| **Основа** | `numpy`, `pandas` | Массивы, таблицы | Необходимо |
| | `scipy` | Статистика, оптимизация | Необходимо |
| | `matplotlib`, `seaborn` | Визуализация | Необходимо |
| **Статистика** | `statsmodels` | Регрессия, тесты, ARIMA | Необходимо |
| | `arch` | GARCH, unit root тесты | Важно |
| | `pmdarima` | Auto-ARIMA | Удобно |
| | `ruptures` | Change-point detection | Полезно |
| | `copulas` | Копулы | Продвинутый |
| **ML** | `scikit-learn` | Классические ML модели | Необходимо |
| | `lightgbm` / `xgboost` | Gradient boosting | Важно |
| | `shap` | Explainability | Важно |
| | `optuna` | Hyperparameter tuning | Удобно |
| | `torch` | Deep learning | По необходимости |
| **Байесовский** | `PyMC` | Байесовский анализ | Продвинутый |
| | `arviz` | Визуализация Bayesian | Продвинутый |
| **Портфель** | `cvxpy` | Convex optimization | Важно |
| | `riskfolio-lib` | Portfolio optimization | Удобно |
| | `pyfolio` | Анализ портфеля | Важно |
| | `empyrical` | Метрики | Удобно |
| **Бэктестинг** | `vectorbt` | Быстрый бэктестинг | Важно |
| | `backtrader` | Event-driven бэктестинг | Альтернатива |
| | `zipline-reloaded` | Quantopian-style | Альтернатива |
| **Данные** | `yfinance` | Yahoo Finance | Бесплатно |
| | `polygon-api-client` | Polygon.io | Платно |
| | `alpaca-trade-api` | Alpaca | Бесплатно |
| | `ccxt` | Крипто биржи (100+) | Бесплатно |
| **Индикаторы** | `ta` | Технические индикаторы | Удобно |
| | `pandas-ta` | TA на pandas | Альтернатива |
| **Временные ряды** | `hmmlearn` | Hidden Markov Models | Полезно |
| | `pywt` | Wavelets | Полезно |
| | `tslearn` | TS clustering/classification | Полезно |
| **NLP** | `transformers` | LLM для sentiment | По необходимости |
| | `vaderSentiment` | Простой sentiment | Быстро |

### 11.2 R (когда Python не хватает)

Некоторые вещи лучше/проще в R:
- `rmgarch` — полноценный DCC-GARCH
- `rugarch` — расширенные GARCH модели
- `PerformanceAnalytics` — метрики портфеля
- `quantmod` — финансовые данные и графики
- `copula` — полный пакет копул

```python
# Можно вызывать R из Python через rpy2
# pip install rpy2
import rpy2.robjects as ro
from rpy2.robjects import pandas2ri
pandas2ri.activate()

ro.r('''
library(rugarch)
spec <- ugarchspec(mean.model=list(armaOrder=c(1,0)),
                   variance.model=list(garchOrder=c(1,1)),
                   distribution.model="sstd")
fit <- ugarchfit(spec, data=returns)
print(fit)
''')
```

### 11.3 Где LLM помогают (расширенная карта)

| Задача | Лучший подход | LLM полезен? |
|--------|---------------|--------------|
| Написать стратегию | Описать логику → LLM генерирует код | Да, очень |
| Найти баг в бэктесте | Показать код → "найди look-ahead bias" | Да, отлично |
| Feature engineering | "Предложи фичи для mean-reversion стратегии" | Да, как генератор идей |
| Sentiment analysis | Claude/GPT API с финансовым промптом | Да, state-of-the-art |
| Объяснить формулу | "Объясни формулу Heston и зачем каждый параметр" | Да, отлично |
| Выбрать метод | "У меня 100 акций, 3 года данных, хочу mean-variance, что учесть?" | Да, хороший советчик |
| Предсказать цену | "Куда пойдёт Bitcoin?" | Нет, LLM не oracle |
| Оптимизация | Числовая оптимизация | Нет, используй scipy/cvxpy |
| Бэктест | Запуск на исторических данных | Нет, используй vectorbt |
| Realtime данные | Стриминг цен | Нет, используй API |

---

## 12. Чеклисты, шпаргалки и war stories

### 12.1 Чеклист: от идеи до live trading

```
═══ ЭТАП 1: ГИПОТЕЗА ═══
□ Сформулирована конкретная тестируемая гипотеза
□ Есть экономическое обоснование (почему это должно работать?)
□ Гипотеза не тривиальна (не "покупай дёшево, продавай дорого")
□ Оценён размер потенциального edge (>0.1% на сделку?)

═══ ЭТАП 2: ДАННЫЕ ═══
□ Достаточно данных (>3 лет для daily, >6 месяцев для intraday)
□ Данные включают разные рыночные режимы (bull, bear, sideways, crisis)
□ Нет survivorship bias (учтены делистинг, банкротства)
□ Point-in-time данные (без пересмотров, доступны были в момент сигнала)
□ Train/validation/test split сделан ДО любого анализа

═══ ЭТАП 3: РЕАЛИЗАЦИЯ ═══
□ Нет look-ahead bias (проверить: будущие данные не используются)
□ Учтены реалистичные задержки (signal → execution gap)
□ Правильная обработка dividends, splits, corporate actions
□ Код проверен на edge cases (пустые данные, gaps, halts)

═══ ЭТАП 4: BACKTEST ═══
□ Реалистичные transaction costs (комиссии + спред + slippage)
□ Учтён market impact (для стратегий с большим объёмом)
□ Time-series cross-validation (не обычный KFold!)
□ Walk-forward analysis проведён
□ Результаты стабильны при ±20% изменении параметров
□ Deflated Sharpe Ratio рассчитан (с учётом числа тестов)

═══ ЭТАП 5: СТАТИСТИЧЕСКАЯ ВАЛИДАЦИЯ ═══
□ Sharpe > 1.0 (годовой, после costs)
□ Max drawdown приемлем для размера капитала
□ Статистически значим (p < 0.05 после multiple testing correction)
□ Bootstrap CI для Sharpe не включает 0
□ Profit factor > 1.5
□ Результат на out-of-sample данных консистентен с in-sample

═══ ЭТАП 6: RISK ═══
□ Position sizing определён (Kelly/fixed fraction)
□ Stop-loss логика реализована и протестирована
□ Стресс-тест на исторических кризисах пройден
□ Корреляция с существующими стратегиями проверена
□ Worst-case сценарий приемлем
□ Есть kill switch (автоматическая остановка при аномалии)

═══ ЭТАП 7: PAPER TRADING ═══
□ Минимум 1-3 месяца paper trading
□ Результаты совпадают с бэктестом (±reasonable variance)
□ Execution quality приемлема (slippage в пределах модели)
□ Инфраструктура стабильна (нет падений, задержек)

═══ ЭТАП 8: LIVE ═══
□ Начинаем с малого размера (10-25% от целевого)
□ Постепенно увеличиваем (если результаты ОК)
□ Мониторинг в реальном времени
□ Alerts на аномалии (необычный drawdown, объём, correlation break)
□ Регулярная ревалидация (ежемесячно/ежеквартально)
```

### 12.2 Формулы — шпаргалка

```
══════════════════════════════════════════════════
ДОХОДНОСТИ
══════════════════════════════════════════════════
Simple return:       r = (P₁ - P₀) / P₀
Log return:          r = ln(P₁ / P₀)
Annualize return:    r_ann = r_daily × 252
Annualize vol:       σ_ann = σ_daily × √252
Geo mean return:     r_geo ≈ r_arith - σ²/2

══════════════════════════════════════════════════
РИСК
══════════════════════════════════════════════════
VaR(α):              Percentile(returns, α)
CVaR(α):             E[r | r < VaR(α)]
VaR (Cornish-Fisher): μ + z_cf × σ, где z_cf = z + (z²-1)s/6 + (z³-3z)k/24
Max Drawdown:        max(peak - value) / peak
Ulcer Index:         √(mean(drawdown²))

══════════════════════════════════════════════════
PERFORMANCE
══════════════════════════════════════════════════
Sharpe:              √252 × mean(r) / std(r)
Sortino:             √252 × mean(r) / downside_std(r)
Calmar:              annual_return / |max_drawdown|
Information Ratio:   √252 × mean(r - r_bench) / std(r - r_bench)
Profit Factor:       Σ(wins) / |Σ(losses)|
Win Rate:            count(r > 0) / count(r)
Payoff Ratio:        mean(wins) / |mean(losses)|
Expectancy:          win_rate × avg_win - loss_rate × avg_loss

══════════════════════════════════════════════════
POSITION SIZING
══════════════════════════════════════════════════
Kelly (single):      f = (p·b - q) / b
Kelly (portfolio):   f = Σ⁻¹(μ - rᶠ)
Half-Kelly:          f / 2
Fixed fraction:      shares = (capital × risk%) / (entry - stop)

══════════════════════════════════════════════════
TIME SERIES
══════════════════════════════════════════════════
OU half-life:        t½ = ln(2) / θ
Hurst exponent:      H via R/S: log(R/S) = H·log(n) + c
GARCH(1,1):          σ²ₜ = ω + α·ε²ₜ₋₁ + β·σ²ₜ₋₁
Persistence:         α + β (close to 1 = persistent vol)
Vol drag:            σ²/2 per year

══════════════════════════════════════════════════
MICROSTRUCTURE
══════════════════════════════════════════════════
Market impact:       I ≈ σ × √(V_trade / V_daily)
Roll spread:         S = 2√(-Cov(rₜ, rₜ₋₁))
Book imbalance:      (bid_vol - ask_vol) / (bid_vol + ask_vol)

══════════════════════════════════════════════════
STATISTICS
══════════════════════════════════════════════════
t-stat for Sharpe:   t = SR × √(n) / √(1 + SR²·(k-1)/4 - SR·s)
Min samples for SR:  n ≈ (z/SR)² × (1 + k/4)
Deflated SR:         P(SR > E[max(SR_i)])
```

### 12.3 War Stories: уроки из реальных провалов

#### LTCM (1998) — Урок о хвостовых рисках и leverage
- **Что делали:** convergence trades с leverage 25:1, модели предполагали нормальные распределения
- **Что пошло не так:** Россия дефолтнула → flight to quality → все спреды разошлись одновременно → correlation spike → margin calls → forced liquidation → спреды разошлись ещё больше (feedback loop)
- **Урок:** (1) Fat tails реальны, (2) корреляции в кризис → 1, (3) leverage убивает, (4) ликвидность исчезает когда нужна больше всего, (5) модель ≠ реальность

#### Knight Capital (2012) — Урок о технических рисках
- **Что произошло:** деплой нового кода, забыли удалить тестовый флаг на одном сервере → алгоритм торговал ПРОТИВ рынка 45 минут
- **Потери:** $440M за 45 минут
- **Урок:** (1) kill switch обязателен, (2) мониторинг позиций в реальном времени, (3) gradual rollout, (4) max loss limits

#### Gaussian Copula и кризис 2008
- **Что делали:** оценивали CDO используя Gaussian copula (Li's formula)
- **Проблема:** Gaussian copula имеет tail dependence = 0 → модель говорила "одновременный дефолт многих заёмщиков практически невозможен"
- **Реальность:** дефолты оказались сильно коррелированы в хвостах → массовые убытки
- **Урок:** (1) tail dependence matters, (2) "все используют эту модель" ≠ "модель правильная", (3) model risk — реальный риск

#### Renaissance Technologies — Урок о том, что работает
- **Medallion Fund:** ~66% годовых (до комиссий) 30+ лет подряд
- **Как (предположительно):** (1) огромное количество слабых сигналов, а не один сильный, (2) короткий holding period (intraday — days), (3) уникальные данные и features, (4) строжайшая статистическая дисциплина, (5) PhD в физике/математике/CS, а не в финансах, (6) инфраструктура на уровне FAANG

### 12.4 Дерево принятия решений

```
У меня есть идея для торговой стратегии. Что дальше?
│
├── Есть экономическое обоснование?
│   ├── Нет → Скорее всего data mining. Будь осторожен.
│   └── Да → Продолжай ↓
│
├── Какой тип стратегии?
│   ├── Mean-reversion
│   │   ├── Один актив → ADF тест, Hurst < 0.5?, OU fit, half-life
│   │   └── Пара/корзина → Коинтеграция, Johansen, спред → OU
│   │
│   ├── Momentum/trend
│   │   ├── Hurst > 0.5?
│   │   ├── Autocorrelation значима?
│   │   └── Breakout vs moving average vs channel
│   │
│   ├── Statistical arbitrage
│   │   ├── PCA → residuals → mean-reversion на residuals
│   │   └── Factor model → alpha = idiosyncratic return
│   │
│   ├── Market making
│   │   ├── Order book data нужны
│   │   ├── Inventory risk management
│   │   └── Adverse selection model
│   │
│   └── Event-driven
│       ├── Earnings, macro releases
│       ├── NLP/LLM для sentiment
│       └── Speed vs depth tradeoff
│
├── Какие данные нужны?
│   ├── Daily OHLCV → yfinance, Polygon (бесплатно)
│   ├── Intraday → Polygon, IEX (платно)
│   ├── Tick/L2 → Специализированные провайдеры ($$$)
│   ├── Fundamental → SEC Edgar (бесплатно), Quandl (платно)
│   ├── Alternative → Web scraping, NLP, satellite (креативность)
│   └── Macro → FRED (бесплатно)
│
├── Какую модель выбрать?
│   ├── Линейные зависимости → Regression, ARIMA
│   ├── Нелинейные → GBM/XGBoost (start here), Neural nets (if needed)
│   ├── Волатильность → GARCH, HAR-RV
│   ├── Режимы → HMM, change-point detection
│   └── Не знаю → Start simple. Linear > tree > neural.
│
└── Как оценить результат?
    ├── Sharpe > 1.0 (after costs)?
    ├── Robust to parameter changes (±20%)?
    ├── Works out-of-sample?
    ├── Deflated Sharpe significant?
    └── Все "да" → Paper trade → Live (small) → Scale up
```

---

## Заключение

### Иерархия важности для алготрейдера

```
1. Риск-менеджмент (без него всё остальное бессмысленно)
2. Статистическая валидация (отличить сигнал от шума)
3. Transaction costs (стратегия прибыльна ПОСЛЕ комиссий?)
4. Простая модель (линейная регрессия, z-score, GARCH)
5. Данные (garbage in → garbage out)
6. Инфраструктура (reliability > speed для большинства)
7. Сложные модели (ML, deep learning) — ТОЛЬКО если простые не хватает
8. Скорость исполнения (важно только для HFT/market making)
```

### Ключевые принципы

1. **Простота.** Начинай с простой модели. Если линейная регрессия даёт Sharpe 0.8, а XGBoost даёт 0.85 — используй линейную (она более robust и понятна).

2. **Статистическая скромность.** Всегда помни: ты можешь обмануть себя. Multiple testing, overfitting, data snooping — враги #1. Deflated Sharpe Ratio — твой друг.

3. **Экономическое обоснование.** Если не можешь объяснить, *почему* стратегия работает (кто теряет деньги, чтобы ты их зарабатывал?) — вероятно, это overfitting.

4. **Волатильность предсказуемее цен.** Это один из немногих надёжных stylized facts. GARCH/HAR-RV для прогнозирования vol работают значительно лучше, чем ARIMA для прогнозирования доходностей.

5. **Диверсификация = единственный бесплатный обед.** Но только если ты правильно моделируешь tail dependence (копулы, стресс-тесты).

6. **Инструменты не заменяют понимание.** Python, ML и LLM — средства. Задавать правильные вопросы и интерпретировать результаты — это работа человека.

### Рекомендуемое чтение

**Must read (топ-5):**
- **Marcos Lopez de Prado, "Advances in Financial Machine Learning"** — как правильно применять ML к финансам, triple barrier, purged CV, meta-labeling, fractional differentiation
- **Ernest Chan, "Quantitative Trading"** — практика алготрейдинга от А до Я
- **Nassim Taleb, "Fooled by Randomness" + "The Black Swan"** — о роли случайности и хвостовых рисках
- **Robert Pardo, "Design, Testing, and Optimization of Trading Systems"** — walk-forward analysis, robustness testing
- **Emanuel Derman, "My Life as a Quant"** — мемуары квант-трейдера, что реально работает

**Продвинутое:**
- **Shreve, "Stochastic Calculus for Finance I & II"** — формальное стохастическое исчисление
- **Cont & Tankov, "Financial Modelling with Jump Processes"** — Lévy процессы
- **Tsay, "Analysis of Financial Time Series"** — полный курс временных рядов для финансов
- **Almgren & Chriss papers** — optimal execution
- **Bouchaud & Potters, "Theory of Financial Risk and Derivative Pricing"** — физик-подход к финансам

**Для CS-инженера:**
- **Stefan Jansen, "Machine Learning for Algorithmic Trading"** — Python-код для всех методов
- **Yves Hilpisch, "Python for Finance"** — Python для финансовых вычислений
- **Max Dama, "The Max Dama Blog"** (blog) — практические заметки HFT-разработчика

---

*Эта книга — карта, а не территория. Каждый раздел — вход в глубокую область. Но для CS-инженера, переходящего в алготрейдинг, этого достаточно чтобы:*
- *Знать, какие инструменты существуют и когда их применять*
- *Не делать грубых ошибок (overfitting, look-ahead bias, игнорирование хвостов)*
- *Разговаривать на одном языке с количественными моделями, библиотеками и LLM*
- *Понимать, где границы каждого метода — и когда нужно копать глубже*
