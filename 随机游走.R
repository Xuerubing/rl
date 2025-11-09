## ========== 配置 ==========
set.seed(1)

N_STATES <- 5                   # 非终止状态个数：1..N_STATES
LEFT_TERM <- 0
RIGHT_TERM <- N_STATES + 1
START_STATE <- (N_STATES + 1) %/% 2   # 中间起点：10
GAMMA <- 1.0
# TRUE_V <- (1:N_STATES) / (N_STATES + 1)   # 真值 i/(N+1)
 # 左侧负一：：
TRUE_V <- (2 * (1:N_STATES) / (N_STATES + 1)) - 1

## ========== 环境 ==========
# 输入: 当前状态 s (0..N+1)
# 输出: list(next_state, reward, done)
step_env <- function(s) {
  if (s == LEFT_TERM || s == RIGHT_TERM) {
    return(list(s, 0.0, TRUE))
  }
  a <- sample(c(-1L, +1L), size = 1L)  # 左右等概率
  s_next <- s + a
  done <- (s_next == LEFT_TERM || s_next == RIGHT_TERM)
  
  # r <- ifelse(s_next == RIGHT_TERM, 1.0, 0.0)
  # 左侧负一：
  r <- ifelse(s_next == RIGHT_TERM,  1.0,
              ifelse(s_next == LEFT_TERM,  -1.0, 0.0))
  
  list(s_next, r, done)
}

gen_episode <- function(start_state = START_STATE) {
  S <- c(start_state)
  R <- c(0.0)  # 对齐索引：R_t 是从 S_{t-1} 到 S_t 的回报
  done <- FALSE
  while (!done) {
    out <- step_env(tail(S, 1))
    S <- c(S, out[[1]])
    R <- c(R, out[[2]])
    done <- out[[3]]
  }
  list(S = S, R = R)  # S_T 为终止状态
}

## ========== n 步 TD 单回合 ==========
# V 长度为 N_STATES+2（含两端终止），索引 0..N+1 通过 +1 偏移映射到 R 下标 1..N+2
n_step_td_episode <- function(V, n = 4L, gamma = GAMMA) {
  ep <- gen_episode(START_STATE)
  S <- ep$S
  R <- ep$R
  Tlen <- length(S) - 1L  # 最后一个 S_T 终止
  
  # 按式(7.2)更新：到期 n 步回报 + bootstrap（未终止时）
  for (t in 0:(Tlen - 1L)) {
    tau <- t - n + 1L
    if (tau < 0L) next
    G <- 0.0
    # 累积真实回报段
    i_end <- min(tau + n, Tlen)
    if (i_end >= tau + 1L) {
      for (i in (tau + 1L):i_end) {
        G <- G + (gamma ^ (i - tau - 1L)) * R[i]
      }
    }
    # bootstrap
    if (tau + n <= Tlen - 1L) {
      s_boot <- S[tau + n + 1L]
      G <- G + (gamma ^ n) * V[s_boot + 1L]
    }
    s_tau <- S[tau + 1L]
    if (s_tau >= 1L && s_tau <= N_STATES) {
      V[s_tau + 1L] <- V[s_tau + 1L] + attr(V, "alpha") * (G - V[s_tau + 1L])
    }
  }
  
  # 尾部 tau = T-n+1,...,T-1（只剩真实回报段，无 bootstrap）
  for (tau in (Tlen - n + 1L):Tlen) {
    if (is.na(tau) || tau < 0L) next
    G <- 0.0
    if (Tlen >= tau + 1L) {
      for (i in (tau + 1L):Tlen) {
        G <- G + (gamma ^ (i - tau - 1L)) * R[i]
      }
    }
    s_tau <- S[tau + 1L]
    if (s_tau >= 1L && s_tau <= N_STATES) {
      V[s_tau + 1L] <- V[s_tau + 1L] + attr(V, "alpha") * (G - V[s_tau + 1L])
    }
  }
  V
}

## ========== 运行多回合/多次独立运行 ==========
init_value <- function(alpha = 0.1) {
  V <- rep(0.0, N_STATES + 2L)  # 含两端
  # V[2:(N_STATES + 1L)] <- 0.5   # 非终止初值 0.5
  # V[LEFT_TERM + 1L] <- 0.0
  # V[RIGHT_TERM + 1L] <- 1.0
  # 改为：
  V[2:(N_STATES + 1L)] <- 0.5     # 非终止初值
  V[LEFT_TERM + 1L] <- -1.0     # 左端
  V[RIGHT_TERM + 1L] <-  1.0    # 右端
  attr(V, "alpha") <- alpha
  V
}

rms_error <- function(V) {
  pred <- V[2:(N_STATES + 1L)]
  sqrt(mean((pred - TRUE_V) ^ 2))
}

run_rw <- function(n = 4L, alpha = 0.1, episodes = 100L, runs = 100L, gamma = GAMMA) {
  rms_mat <- matrix(0.0, nrow = runs, ncol = episodes)
  for (r in 1:runs) {
    V <- init_value(alpha)
    for (ep in 1:episodes) {
      V <- n_step_td_episode(V, n = n, gamma = gamma)
      rms_mat[r, ep] <- rms_error(V)
    }
  }
  colMeans(rms_mat)
}

## ========== 图 1：固定 alpha，比较不同 n 的 RMS vs Episodes ==========
plot_rms_vs_episodes <- function(n_list = c( 2, 4, 8),
                                 alpha = 0.1, episodes = 100L, runs = 200L) {
  res <- lapply(n_list, function(n) run_rw(n = n, alpha = alpha,
                                           episodes = episodes, runs = runs))
  plot(1:episodes, res[[1]], type = "l", lwd = 2,
       xlab = "Episodes", ylab = "Average RMS error",
       main = sprintf("n-step TD on %d-State Random Walk (alpha=%.3f, runs=%d)",
                      N_STATES, alpha, runs))
  cols <- c("black", "red", "blue", "darkgreen", "purple", "orange")
  for (i in seq_along(n_list)) {
    lines(1:episodes, res[[i]], lwd = 2, col = cols[i])
  }
  legend("topright", legend = paste0("n=", n_list),
         col = cols[seq_along(n_list)], lwd = 2, cex = 0.9)
}

## ========== 图 2：固定 Episodes=10，扫 alpha，比较不同 n 的 RMS vs alpha ==========
sweep_alpha <- function(n_list = c(2, 4, 8),
                        alphas = seq(0.05, 1.0, by = 0.05),
                        episodes = 10L, runs = 1000L) {
  avg_rms <- matrix(NA_real_, nrow = length(n_list), ncol = length(alphas))
  for (i in seq_along(n_list)) {
    for (j in seq_along(alphas)) {
      avg_rms[i, j] <- tail(run_rw(n = n_list[i], alpha = alphas[j],
                                   episodes = episodes, runs = runs), 1L)
    }
  }
  list(n_list = n_list, alphas = alphas, avg_rms = avg_rms)
}

plot_rms_vs_alpha <- function(sweep_obj) {
  n_list <- sweep_obj$n_list
  alphas <- sweep_obj$alphas
  avg_rms <- sweep_obj$avg_rms
  
  ylim <- range(avg_rms, na.rm = TRUE)
  plot(alphas, avg_rms[1, ], type = "l", lwd = 2, ylim = ylim,
       xlab = expression(alpha), ylab = "Average RMS @ episode=10",
       main = sprintf("RMS vs alpha on %d-State Random Walk", N_STATES))
  cols <- c("black", "red", "blue")
  # cols <- c("black", "red", "blue", "darkgreen", "purple", "orange")
  for (i in seq_along(n_list)) {
    lines(alphas, avg_rms[i, ], lwd = 2, col = cols[i])
  }
  legend("topright", legend = paste0("n=", n_list),
         col = cols[seq_along(n_list)], lwd = 2, cex = 0.9)
}

## ========== Demo ==========
# 1) 固定 alpha 画 RMS vs Episodes
plot_rms_vs_episodes(n_list = c(2, 4, 8),
                     alpha = 0.1, episodes = 100L, runs = 200L)

# 2) 扫描 alpha 画 RMS vs alpha（默认 episodes=10）
swp <- sweep_alpha(n_list = c(2, 4, 8),
                   alphas = seq(0.05, 1.0, by = 0.05),
                   episodes = 10L, runs = 1000L)
plot_rms_vs_alpha(swp)
