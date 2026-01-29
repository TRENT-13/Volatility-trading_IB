# =============================================================================
# IMPORTS
# =============================================================================
import tkinter as tk                                    # GUI framework
from tkinter import ttk, messagebox                     # Themed widgets and dialogs
import threading                                        # For background threads (IB connection, bar management)
import time                                             # Sleep and timing functions
from collections import deque                           # Fixed-size queue for rolling data
from datetime import datetime                           # Timestamps for bars
import numpy as np                                      # Numerical operations for regime model
from scipy import stats                                 # Statistical functions for hypothesis testing
import matplotlib.pyplot as plt                         # Plotting library
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg  # Embed matplotlib in tkinter
from matplotlib.patches import Rectangle               # Draw candlestick bodies and regime backgrounds
import matplotlib.dates as mdates                       # Date formatting for axes
from ibapi.client import EClient                        # IB API client for sending requests
from ibapi.wrapper import EWrapper                      # IB API wrapper for receiving responses
from ibapi.contract import Contract                     # Define financial instruments
from ibapi.order import Order                           # Order objects for trading
import queue                                            # Thread-safe queue for tick data
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")


# =============================================================================
# INTERACTIVE BROKERS API WRAPPER
# =============================================================================
class IBApp(EWrapper, EClient):
    """Handles communication with TWS - inherits from both EWrapper (callbacks) and EClient (requests)."""

    def __init__(self, callback=None):
        EClient.__init__(self, self)
        self.connected = False                          # Connection status flag
        self.callback = callback                        # Function to call when tick data arrives
        self.last_price = None                          # Most recent trade price
        self.bid_price = None                           # Current bid price
        self.ask_price = None                           # Current ask price
        self.historical_data = {}                       # Store historical bars by request ID
        self.hist_done = threading.Event()              # Signal when historical data is complete
        # Add after self.streaming = False
        self.simulation_mode = False
        self.simulation_data = []
        self.simulation_index = 0
        self.playback_speed = 1.0

    def error(self, reqId, errorCode, errorString, advancedOrderRejectJson=""):
        # Ignore common non-error messages (connection confirmations, data farm status)
        if errorCode in [2104, 2106, 2158, 2176]:
            return
        # Note when using delayed (free) market data instead of real-time
        if errorCode == 10167:
            print(f"Note: Using delayed market data")
            return
        print(f"Error {reqId}: {errorCode} - {errorString}")

    def nextValidId(self, orderId):
        # Called when connection is established - TWS sends first valid order ID
        self.connected = True
        self.next_order_id = orderId
        print(f"Connected to TWS. Next valid order ID: {orderId}")

    def historicalData(self, reqId, bar):
        # Called for each bar of historical data - store OHLC values
        if reqId not in self.historical_data:
            self.historical_data[reqId] = []
        self.historical_data[reqId].append({'o': bar.open, 'h': bar.high, 'l': bar.low, 'c': bar.close})

    def historicalDataEnd(self, reqId, start, end):
        # Called when all historical data has been received - signal completion
        self.hist_done.set()

    def tickPrice(self, reqId, tickType, price, attrib):
        # Called when a price tick arrives from live market data stream
        if price <= 0:
            return
        # tickType codes: 1=Bid, 2=Ask, 4=Last, 6=High, 7=Low, 9=Close
        if tickType == 4:                               # Last traded price
            self.last_price = price
            if self.callback:
                self.callback('price', price, datetime.now())
        elif tickType == 1:                             # Bid price
            self.bid_price = price
        elif tickType == 2:                             # Ask price
            self.ask_price = price

    def tickSize(self, reqId, tickType, size):
        # Called for volume/size ticks - not used in this application
        pass

    def tickString(self, reqId, tickType, value):
        # Called for string tick data (e.g., last trade time) - not used
        pass

    def orderStatus(self, orderId, status, filled, remaining, avgFillPrice, 
                   permId, parentId, lastFillPrice, clientId, whyHeld, mktCapPrice):
        # Called when order status changes
        print(f"Order {orderId}: Status={status}, Filled={filled}, AvgPrice={avgFillPrice:.2f}")
        if hasattr(self, 'order_callback') and self.order_callback:
            self.order_callback(orderId, status, filled, avgFillPrice)

    def execDetails(self, reqId, contract, execution):
        # Called when an order is executed
        print(f"Execution: {execution.side} {execution.shares} {contract.symbol} @ {execution.price:.2f}")

    def openOrder(self, orderId, contract, order, orderState):
        # Called for open order information
        print(f"Open Order {orderId}: {order.action} {order.totalQuantity} {contract.symbol}")


# =============================================================================
# OHLC BAR DATA STRUCTURE
# =============================================================================
class OHLCBar:
    """Represents a single OHLC (Open-High-Low-Close) candlestick bar."""
    
    def __init__(self, timestamp, open_price):
        self.timestamp = timestamp                      # When bar started
        self.open = open_price                          # First price in bar
        self.high = open_price                          # Highest price seen
        self.low = open_price                           # Lowest price seen
        self.close = open_price                         # Most recent price (updates with each tick)
        self.tick_count = 1                             # Number of ticks in this bar
        self.regime = 0                                 # Volatility regime: 0=low, 1=med, 2=high

    def update(self, price):
        # Update bar with new tick - adjust high/low/close accordingly
        self.high = max(self.high, price)
        self.low = min(self.low, price)
        self.close = price
        self.tick_count += 1

    @property
    def volatility(self):
        # Bar volatility = price range as percentage of close price
        return (self.high - self.low) / self.close if self.close > 0 else 0


# =============================================================================
# MARKOV REGIME MODEL
# =============================================================================
# A Hidden Markov Model (HMM) treats the market regime as a "hidden" state that
# we cannot directly observe. Instead, we observe volatility and must infer
# which regime we're in based on:
#   1. The TRANSITION MATRIX: How likely is it to switch between regimes?
#   2. The EMISSION MODEL: What volatility do we expect to see in each regime?
# 
# The model works by maintaining a probability distribution over regimes and
# updating it each time we observe a new bar's volatility using Bayesian inference.
# =============================================================================
class MarkovRegime:
    """
    3-state Markov Chain Regime Switching model for volatility classification.
    
    Uses a transition probability matrix and Gaussian emission distributions to
    compute the posterior probability of each regime given observed volatility.
    The regime is determined by the highest likelihood of observing the realized
    bar volatility, weighted by transition probabilities from the previous state.
    
    IMPORTANT: All parameters (transition matrix, emission means/stds) are 
    ESTIMATED FROM DATA using Maximum Likelihood Estimation (MLE), not hardcoded.
    """
    
    def __init__(self):
        # =====================================================================
        # MODEL CONFIGURATION
        # =====================================================================
        # We define 3 volatility regimes:
        #   State 0 = LOW volatility  (calm market, small price movements)
        #   State 1 = MED volatility  (normal market conditions)
        #   State 2 = HIGH volatility (turbulent market, large price swings)
        self.n_states = 3
        self.current_state = 0                          # Current most likely regime
        self.colors = ['#3fb950', '#d29922', '#f85149'] # Regime label colors: green, orange, red
        self.bg_colors = ['#1a3d1a', '#3d3319', '#3d1a1a']  # Muted background colors for chart
        
        # =====================================================================
        # STATE PROBABILITY VECTOR (aka "belief state" or "filtered distribution")
        # =====================================================================
        # This vector holds our current belief about which regime we're in:
        #   state_probs[i] = P(regime = i | all observations so far)
        # 
        # At any time, these probabilities sum to 1. For example:
        #   [0.7, 0.2, 0.1] means 70% chance LOW, 20% MED, 10% HIGH
        # 
        # We start with uniform (equal) probabilities since we have no information yet.
        self.state_probs = np.array([1/3, 1/3, 1/3])
        
        # =====================================================================
        # TRANSITION PROBABILITY MATRIX (the "Markov" part)
        # =====================================================================
        # The transition matrix captures the MARKOV PROPERTY: the probability of
        # the next state depends ONLY on the current state, not on history.
        # 
        # T[i,j] = P(next_regime = j | current_regime = i)
        # 
        # THESE ARE INITIAL VALUES ONLY - they will be replaced by MLE estimates
        # from historical data during calibration. We use uniform priors initially.
        # 
        # Initial assumption: slight preference for staying in same state (ergodic)
        # This ensures the Markov chain is well-defined before calibration.
        self.transition_matrix = np.array([
            # To:    LOW   MED   HIGH
            [1/3, 1/3, 1/3],  # Uniform prior - will be estimated from data
            [1/3, 1/3, 1/3],  # Uniform prior - will be estimated from data
            [1/3, 1/3, 1/3]   # Uniform prior - will be estimated from data
        ])
        
        # =====================================================================
        # EMISSION DISTRIBUTION PARAMETERS (the "Hidden" part)
        # =====================================================================
        # Each regime "emits" volatility values according to a Gaussian distribution.
        # This is the EMISSION MODEL: P(observed_volatility | regime)
        # 
        # THESE ARE PLACEHOLDER VALUES - they will be estimated from data using MLE.
        # Initial values are spaced to ensure numerical stability before calibration.
        self.emission_means = np.array([0.001, 0.003, 0.006])  # Placeholder
        self.emission_stds = np.array([0.0005, 0.001, 0.002])  # Placeholder
        
        # =====================================================================
        # CALIBRATION DIAGNOSTICS
        # =====================================================================
        # Store information about the calibration for statistical validation
        self.is_calibrated = False
        self.calibration_stats = {
            'n_samples': 0,
            'transition_counts': None,
            'regime_counts': None,
            'log_likelihood': None,
            'aic': None,  # Akaike Information Criterion
            'bic': None   # Bayesian Information Criterion
        }

    def calibrate(self, hist_bars):
        """
        Calibrate all model parameters from historical data using Maximum Likelihood Estimation.
        
        This method implements proper statistical estimation:
        1. EMISSION PARAMETERS: Estimated using MLE for Gaussian mixture
        2. TRANSITION MATRIX: Estimated using MLE with Laplace smoothing
        3. MODEL VALIDATION: Computes AIC/BIC for model selection
        
        The approach:
        - First, assign bars to regimes using percentile-based clustering
        - Then, estimate emission parameters (μ, σ) for each regime using MLE
        - Finally, estimate transition probabilities using frequency counting (MLE)
        
        Mathematical Foundation:
        - For emission parameters: MLE of Gaussian gives μ̂ = sample mean, σ̂ = sample std
        - For transition matrix: MLE gives T̂[i,j] = N[i,j] / Σⱼ N[i,j]
          where N[i,j] is the count of transitions from state i to state j
        """
        # Need enough data for reliable estimation (rule of thumb: 10+ per parameter)
        if len(hist_bars) < 30:
            print(f"Warning: Insufficient data for calibration ({len(hist_bars)} bars, need 30+)")
            return
        
        # =====================================================================
        # STEP 1: COMPUTE HISTORICAL VOLATILITIES
        # =====================================================================
        # Volatility = (High - Low) / Close for each bar (True Range approximation)
        # This measures the price range as a fraction of the closing price
        vols = np.array([(b['h'] - b['l']) / b['c'] if b['c'] > 0 else 0 for b in hist_bars])
        vols = vols[vols > 0]  # Remove zero volatility bars (no price movement)
        
        if len(vols) < 30:
            print(f"Warning: Insufficient non-zero volatility samples ({len(vols)})")
            return
        
        # =====================================================================
        # STEP 2: REGIME ASSIGNMENT USING PERCENTILE CLUSTERING
        # =====================================================================
        # We use percentile-based assignment as an initial clustering step.
        # This is justified because:
        #   - Volatility regimes are typically defined by relative levels
        #   - Percentiles are robust to outliers
        #   - This is equivalent to empirical quantile-based discretization
        #
        # Statistical justification: Under the assumption that each regime
        # contributes roughly equally to the data, percentiles give consistent
        # estimates of regime boundaries.
        p33, p67 = np.percentile(vols, 33), np.percentile(vols, 67)
        
        # Create array of regime assignments (0, 1, or 2 for each bar)
        regime_assignments = np.zeros(len(vols), dtype=int)
        regime_assignments[vols >= p33] = 1   # MED if above 33rd percentile
        regime_assignments[vols >= p67] = 2   # HIGH if above 67th percentile
        
        # =====================================================================
        # STEP 3: MLE FOR EMISSION PARAMETERS (Gaussian parameters per regime)
        # =====================================================================
        # For each regime, we estimate the Gaussian parameters using MLE:
        #   μ̂ (MLE) = (1/n) Σ xᵢ  (sample mean)
        #   σ̂ (MLE) = √[(1/n) Σ (xᵢ - μ̂)²]  (sample standard deviation)
        #
        # Note: We use n instead of (n-1) for true MLE, though the difference
        # is negligible for our sample sizes.
        
        regime_counts = np.zeros(self.n_states)
        temp_means = np.zeros(self.n_states)
        temp_stds = np.zeros(self.n_states)
        
        for regime in range(self.n_states):
            # Get all volatilities assigned to this regime
            regime_vols = vols[regime_assignments == regime]
            regime_counts[regime] = len(regime_vols)
            
            if len(regime_vols) >= 5:  # Need minimum samples for reliable estimation
                # MLE for Gaussian mean
                temp_means[regime] = np.mean(regime_vols)
                # MLE for Gaussian std (using n, not n-1)
                # Add small constant for numerical stability
                temp_stds[regime] = max(np.std(regime_vols, ddof=0), 1e-6)
            else:
                # Fallback: interpolate from neighboring regimes or use defaults
                print(f"Warning: Regime {regime} has only {len(regime_vols)} samples")
                temp_means[regime] = np.percentile(vols, 33 * regime + 16.5)
                temp_stds[regime] = np.std(vols) / 3
        
        # Ensure means are properly ordered: LOW < MED < HIGH
        # This is a constraint that must be satisfied for interpretability
        sorted_indices = np.argsort(temp_means)
        self.emission_means = temp_means[sorted_indices]
        self.emission_stds = temp_stds[sorted_indices]
        
        # Re-map regime assignments if order changed
        if not np.array_equal(sorted_indices, np.arange(self.n_states)):
            # Create mapping from old indices to new
            index_map = {old: new for new, old in enumerate(sorted_indices)}
            regime_assignments = np.array([index_map[r] for r in regime_assignments])
        
        # =====================================================================
        # STEP 4: MLE FOR TRANSITION MATRIX
        # =====================================================================
        # The MLE for transition probabilities is simply the frequency estimate:
        #   T̂[i,j] = N[i,j] / N[i,•]
        # where N[i,j] = count of transitions from state i to j
        #       N[i,•] = total transitions out of state i
        #
        # We apply Laplace (add-1) smoothing to handle zero counts:
        #   T̂[i,j] = (N[i,j] + α) / (N[i,•] + α*K)
        # where α = smoothing parameter (we use 1.0 for standard Laplace)
        #       K = number of states
        #
        # This corresponds to a Bayesian estimate with uniform Dirichlet prior.
        
        transition_counts = np.zeros((self.n_states, self.n_states))
        
        # Count transitions in the observed sequence
        for t in range(1, len(regime_assignments)):
            prev_regime = regime_assignments[t-1]  # Where we were (Sₜ₋₁)
            curr_regime = regime_assignments[t]     # Where we went (Sₜ)
            transition_counts[prev_regime, curr_regime] += 1
        
        # Apply MLE with Laplace smoothing (α = 1.0)
        alpha = 1.0  # Smoothing parameter (Laplace/add-1 smoothing)
        for i in range(self.n_states):
            row_sum = transition_counts[i].sum()
            if row_sum > 0:
                # MLE with Laplace smoothing: (count + α) / (total + α*K)
                self.transition_matrix[i] = (transition_counts[i] + alpha) / (row_sum + alpha * self.n_states)
            else:
                # No transitions observed from this state - use uniform
                self.transition_matrix[i] = np.ones(self.n_states) / self.n_states
        
        # =====================================================================
        # STEP 5: COMPUTE MODEL FIT STATISTICS (for validation)
        # =====================================================================
        # Log-likelihood of the fitted model
        log_likelihood = self._compute_log_likelihood(vols, regime_assignments)
        
        # Number of free parameters:
        # - Emission: 2 params per state (mean, std) = 6
        # - Transition: (K-1)*K params (each row sums to 1) = 6
        # Total: 12 parameters
        n_params = 2 * self.n_states + self.n_states * (self.n_states - 1)
        n_samples = len(vols)
        
        # Akaike Information Criterion: AIC = 2k - 2ln(L)
        # Lower is better; penalizes model complexity
        aic = 2 * n_params - 2 * log_likelihood
        
        # Bayesian Information Criterion: BIC = k*ln(n) - 2ln(L)
        # Stronger penalty for complexity than AIC
        bic = n_params * np.log(n_samples) - 2 * log_likelihood
        
        # Store calibration diagnostics
        self.is_calibrated = True
        self.calibration_stats = {
            'n_samples': n_samples,
            'transition_counts': transition_counts.copy(),
            'regime_counts': regime_counts.copy(),
            'log_likelihood': log_likelihood,
            'aic': aic,
            'bic': bic
        }
        
        # Compute stationary distribution (long-run regime probabilities)
        stationary_dist = self._compute_stationary_distribution()
        
        # Reset state probabilities to stationary distribution
        self.state_probs = stationary_dist
        
        # Print calibration summary
        print(f"\n{'='*60}")
        print(f"REGIME MODEL CALIBRATION COMPLETE")
        print(f"{'='*60}")
        print(f"Samples used: {n_samples}")
        print(f"\nEmission Parameters (MLE estimates):")
        print(f"  LOW:  μ = {self.emission_means[0]:.6f}, σ = {self.emission_stds[0]:.6f}")
        print(f"  MED:  μ = {self.emission_means[1]:.6f}, σ = {self.emission_stds[1]:.6f}")
        print(f"  HIGH: μ = {self.emission_means[2]:.6f}, σ = {self.emission_stds[2]:.6f}")
        print(f"\nTransition Matrix (MLE with Laplace smoothing):")
        print(f"        To LOW    To MED    To HIGH")
        for i, name in enumerate(['LOW ', 'MED ', 'HIGH']):
            print(f"  {name}  [{self.transition_matrix[i,0]:.4f}]  [{self.transition_matrix[i,1]:.4f}]  [{self.transition_matrix[i,2]:.4f}]")
        print(f"\nStationary Distribution: {stationary_dist}")
        print(f"\nModel Fit Statistics:")
        print(f"  Log-Likelihood: {log_likelihood:.2f}")
        print(f"  AIC: {aic:.2f}")
        print(f"  BIC: {bic:.2f}")
        print(f"{'='*60}\n")
    
    def _compute_log_likelihood(self, vols, regime_assignments):
        """
        Compute the log-likelihood of the data under the fitted model.
        
        L(θ|data) = Π P(vₜ|regime_t) × P(regime_t|regime_{t-1})
        log L = Σ [log P(vₜ|regime_t) + log P(regime_t|regime_{t-1})]
        """
        log_lik = 0.0
        
        for t in range(len(vols)):
            regime = regime_assignments[t]
            
            # Emission probability: P(vol|regime) using Gaussian PDF
            emission_prob = stats.norm.pdf(vols[t], 
                                           self.emission_means[regime], 
                                           self.emission_stds[regime])
            if emission_prob > 0:
                log_lik += np.log(emission_prob)
            
            # Transition probability: P(regime_t|regime_{t-1})
            if t > 0:
                prev_regime = regime_assignments[t-1]
                trans_prob = self.transition_matrix[prev_regime, regime]
                if trans_prob > 0:
                    log_lik += np.log(trans_prob)
        
        return log_lik
    
    def _compute_stationary_distribution(self):
        """
        Compute the stationary distribution of the Markov chain.
        
        The stationary distribution π satisfies: π = π × T
        It represents the long-run probability of being in each state.
        
        We find it by solving the eigenvector problem: eigenvalue = 1.
        """
        # Find eigenvalues and eigenvectors of T^T
        eigenvalues, eigenvectors = np.linalg.eig(self.transition_matrix.T)
        
        # Find the eigenvector corresponding to eigenvalue 1
        # (there should always be one for a valid stochastic matrix)
        idx = np.argmin(np.abs(eigenvalues - 1))
        stationary = np.real(eigenvectors[:, idx])
        
        # Normalize to get probabilities (sum to 1)
        stationary = stationary / stationary.sum()
        
        # Ensure all positive (might have small numerical errors)
        stationary = np.abs(stationary)
        stationary = stationary / stationary.sum()
        
        return stationary

    def _gaussian_likelihood(self, vol, regime):
        """
        Compute the probability density of observing volatility 'vol' given 'regime'.
        Uses Gaussian (normal) distribution as the emission model.
        
        This answers: "If we're in this regime, how likely is it to see this volatility?"
        """
        mean = self.emission_means[regime]  # Expected volatility for this regime
        std = self.emission_stds[regime]    # Spread of volatility for this regime
        
        # =====================================================================
        # GAUSSIAN PROBABILITY DENSITY FUNCTION (PDF)
        # =====================================================================
        # Formula: P(x) = (1 / (σ * √(2π))) * exp(-0.5 * ((x - μ) / σ)²)
        # 
        # Where:
        #   x = observed volatility (vol)
        #   μ = mean (expected volatility for regime)
        #   σ = standard deviation (spread)
        # 
        # The PDF gives a higher value when vol is close to the mean,
        # and lower values as vol moves away from the mean.
        coeff = 1 / (std * np.sqrt(2 * np.pi))  # Normalization constant
        exponent = -0.5 * ((vol - mean) / std) ** 2  # Squared distance from mean
        return coeff * np.exp(exponent)
        # 
        # Example: If LOW regime has mean=0.001 and we observe vol=0.001,
        # the likelihood is high. If we observe vol=0.01, likelihood is very low.

    def get_regime(self, bars):
        """
        Determine the current regime using the Markov Chain filtering approach.
        
        This is the core inference algorithm that uses Bayes' rule to update
        our belief about which regime we're in based on observed volatility.
        """
        if not bars:
            return self.current_state
        
        # Get the most recent bar's volatility - this is our new observation
        current_bar = bars[-1]
        vol = current_bar.volatility  # (high - low) / close
        
        if vol <= 0:
            current_bar.regime = self.current_state
            return self.current_state
        
        # =====================================================================
        # STEP 1: PREDICTION STEP (Time Update)
        # =====================================================================
        # Before seeing the new observation, predict where we might be based
        # on where we were and the transition probabilities.
        # 
        # Math: prior[j] = Σᵢ T[i,j] * state_probs[i]
        # "Sum over all possible previous states, weighted by their probability"
        # 
        # In matrix form: prior = T^T @ state_probs
        # (T^T is transpose because we want column j from each row)
        prior_probs = self.transition_matrix.T @ self.state_probs
        # 
        # Example: If state_probs = [0.8, 0.15, 0.05] (80% sure we're in LOW),
        # then prior might be [0.75, 0.18, 0.07] after applying transition probs.
        
        # =====================================================================
        # STEP 2: COMPUTE EMISSION LIKELIHOODS
        # =====================================================================
        # For each regime, compute: "How likely is this volatility if we're in that regime?"
        # 
        # likelihoods[i] = P(vol | regime = i)
        # 
        # This uses the Gaussian PDF for each regime's emission distribution.
        likelihoods = np.array([self._gaussian_likelihood(vol, i) for i in range(self.n_states)])
        # 
        # Example: If vol=0.001 (low volatility):
        #   likelihoods might be [0.95, 0.30, 0.05]
        #   HIGH likelihood for LOW regime, low likelihood for HIGH regime
        
        # =====================================================================
        # STEP 3: UPDATE STEP (Measurement Update) - BAYES' RULE
        # =====================================================================
        # Combine the prior (where transition matrix says we should be) with
        # the likelihood (what the observed volatility suggests).
        # 
        # Bayes' Rule: P(regime | vol) ∝ P(vol | regime) × P(regime)
        #                posterior    ∝   likelihood    ×   prior
        # 
        # The regime with high prior AND high likelihood wins.
        posterior_probs = prior_probs * likelihoods
        # 
        # Example: If prior = [0.75, 0.18, 0.07] and likelihoods = [0.95, 0.30, 0.05]
        # Then posterior ∝ [0.75*0.95, 0.18*0.30, 0.07*0.05] = [0.7125, 0.054, 0.0035]
        
        # Normalize so probabilities sum to 1
        # This is the denominator in Bayes' rule: P(vol) = Σᵢ P(vol|i)P(i)
        prob_sum = posterior_probs.sum()
        if prob_sum > 0:
            posterior_probs = posterior_probs / prob_sum
        else:
            # Fallback to prior if all likelihoods are zero (shouldn't happen)
            posterior_probs = prior_probs
        # 
        # After normalization: [0.926, 0.070, 0.004] = 92.6% sure we're in LOW
        
        # Save updated belief for next iteration (this is the "filtering" part)
        # Our belief state carries forward through time
        self.state_probs = posterior_probs
        
        # =====================================================================
        # STEP 4: SELECT MOST LIKELY REGIME (Maximum A Posteriori - MAP)
        # =====================================================================
        # Choose the regime with the highest posterior probability
        # This is our best guess given all the evidence
        self.current_state = int(np.argmax(posterior_probs))
        current_bar.regime = self.current_state
        # 
        # The regime with highest probability "wins" - this is what we display
        # on the chart and use for trading decisions
        
        return self.current_state


# =============================================================================
# HMM-BASED TRADING STRATEGY
# =============================================================================
# This trading strategy uses statistical concepts from the HMM to generate
# trading signals with proper hypothesis testing and confidence measures.
# =============================================================================
class TradingStrategy:
    """
    Enhanced trading strategy based on HMM regime switching with statistical validation.
    
    Uses the following statistical concepts:
    1. Maximum Likelihood Estimation (MLE) - for regime parameter calibration
    2. Maximum A Posteriori (MAP) - for regime classification
    3. Hypothesis Testing - to validate regime changes are significant
    4. P-values - to measure statistical significance of signals
    5. Trend Detection - using linear regression slope with t-test
    6. Momentum Indicators - rate of change and acceleration
    7. Mean Reversion Detection - distance from rolling mean in std units
    """
    
    def __init__(self):
        # Position tracking
        self.position = 0                               # Current position: +1=long, -1=short, 0=flat
        self.position_price = 0.0                       # Entry price
        self.position_size = 10                         # Number of shares per trade
        
        # Trade history for performance tracking
        self.trades = []                                # List of completed trades
        self.signals = []                               # List of all signals generated
        
        # Strategy parameters
        self.confidence_threshold = 0.70                # Minimum MAP probability to act (70%)
        self.significance_level = 0.05                  # Alpha for hypothesis tests (5%)
        self.min_bars_for_signal = 5                    # Minimum bars before generating signals
        self.regime_history = []                        # Track regime changes
        self.volatility_history = []                    # Track volatility observations
        
        # =====================================================================
        # TREND AND MOMENTUM TRACKING
        # =====================================================================
        self.price_history = []                         # Historical close prices
        self.return_history = []                        # Historical returns
        
        # Trend parameters (for linear regression)
        self.trend_window = 10                          # Lookback for trend calculation
        self.trend_slope = 0.0                          # Current trend slope
        self.trend_r_squared = 0.0                      # R² of trend fit
        self.trend_p_value = 1.0                        # P-value for trend significance
        
        # Momentum parameters
        self.momentum_window = 5                        # Lookback for momentum
        self.momentum = 0.0                             # Current momentum (rate of change)
        self.acceleration = 0.0                         # Change in momentum
        
        # Mean reversion parameters
        self.mean_reversion_window = 20                 # Lookback for mean calculation
        self.z_score = 0.0                              # Distance from mean in std units
        self.mean_reversion_threshold = 2.0            # Z-score threshold for mean reversion
        
        # P&L tracking
        self.realized_pnl = 0.0
        self.unrealized_pnl = 0.0
    
    # =========================================================================
    # TREND ANALYSIS METHODS
    # =========================================================================
    
    def compute_trend(self, prices):
        """
        Compute trend using Ordinary Least Squares (OLS) linear regression.
        
        Model: price_t = α + β*t + ε
        
        We estimate β (slope) and test if it's significantly different from 0.
        H₀: β = 0 (no trend)
        H₁: β ≠ 0 (trend exists)
        
        Returns: (slope, r_squared, p_value)
        """
        if len(prices) < 3:
            return 0.0, 0.0, 1.0
        
        n = len(prices)
        x = np.arange(n)  # Time index: 0, 1, 2, ..., n-1
        y = np.array(prices)
        
        # Normalize prices for numerical stability
        y_mean = np.mean(y)
        y_normalized = (y - y_mean) / y_mean if y_mean != 0 else y
        
        # OLS: β̂ = Σ(x - x̄)(y - ȳ) / Σ(x - x̄)²
        x_mean = np.mean(x)
        y_norm_mean = np.mean(y_normalized)
        
        numerator = np.sum((x - x_mean) * (y_normalized - y_norm_mean))
        denominator = np.sum((x - x_mean) ** 2)
        
        if denominator == 0:
            return 0.0, 0.0, 1.0
        
        slope = numerator / denominator
        intercept = y_norm_mean - slope * x_mean
        
        # Predicted values and residuals
        y_pred = intercept + slope * x
        residuals = y_normalized - y_pred
        
        # R-squared: proportion of variance explained
        ss_res = np.sum(residuals ** 2)
        ss_tot = np.sum((y_normalized - y_norm_mean) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        # T-test for slope significance
        # Standard error of slope: SE(β̂) = s / √Σ(x - x̄)²
        # where s² = Σresiduals² / (n-2)
        if n > 2:
            mse = ss_res / (n - 2)
            se_slope = np.sqrt(mse / denominator) if mse > 0 else 1e-10
            t_stat = slope / se_slope if se_slope > 0 else 0
            # Two-tailed p-value
            p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df=n-2))
        else:
            p_value = 1.0
        
        # Store for later use
        self.trend_slope = slope
        self.trend_r_squared = r_squared
        self.trend_p_value = p_value
        
        return slope, r_squared, p_value
    
    def compute_momentum(self, prices):
        """
        Compute momentum as the rate of change (ROC) and acceleration.
        
        Momentum (ROC) = (P_t - P_{t-n}) / P_{t-n}
        Acceleration = Momentum_t - Momentum_{t-1}
        
        Momentum measures the velocity of price changes.
        Acceleration measures if momentum is increasing or decreasing.
        """
        if len(prices) < self.momentum_window + 1:
            return 0.0, 0.0
        
        # Rate of Change (momentum)
        current_price = prices[-1]
        past_price = prices[-self.momentum_window]
        momentum = (current_price - past_price) / past_price if past_price != 0 else 0
        
        # Acceleration (change in momentum)
        if len(prices) >= self.momentum_window + 2:
            prev_price = prices[-2]
            prev_past_price = prices[-self.momentum_window - 1]
            prev_momentum = (prev_price - prev_past_price) / prev_past_price if prev_past_price != 0 else 0
            acceleration = momentum - prev_momentum
        else:
            acceleration = 0.0
        
        self.momentum = momentum
        self.acceleration = acceleration
        
        return momentum, acceleration
    
    def compute_z_score(self, prices):
        """
        Compute Z-score: how many standard deviations the current price is from the mean.
        
        Z = (P_t - μ) / σ
        
        where μ = rolling mean over window
              σ = rolling standard deviation
        
        Used for mean reversion signals:
        - Z > 2: Price is unusually high, might revert down
        - Z < -2: Price is unusually low, might revert up
        """
        if len(prices) < self.mean_reversion_window:
            return 0.0
        
        window_prices = prices[-self.mean_reversion_window:]
        mean = np.mean(window_prices)
        std = np.std(window_prices)
        
        if std == 0:
            return 0.0
        
        z_score = (prices[-1] - mean) / std
        self.z_score = z_score
        
        return z_score
    
    def get_trend_direction(self):
        """
        Classify trend direction with statistical significance.
        
        Returns: ('UP', 'DOWN', 'NEUTRAL') with confidence level
        """
        if self.trend_p_value > self.significance_level:
            # Trend is not statistically significant
            return 'NEUTRAL', 1 - self.trend_p_value
        
        if self.trend_slope > 0:
            return 'UP', 1 - self.trend_p_value
        else:
            return 'DOWN', 1 - self.trend_p_value
    
    def get_momentum_signal(self):
        """
        Generate momentum-based signal component.
        
        Returns: (signal_strength, direction)
        signal_strength: 0 to 1 (1 = strong signal)
        direction: 'BULLISH', 'BEARISH', 'NEUTRAL'
        """
        # Strong positive momentum with positive acceleration = bullish
        # Strong negative momentum with negative acceleration = bearish
        
        if abs(self.momentum) < 0.001:  # Very small momentum
            return 0.0, 'NEUTRAL'
        
        # Signal strength based on momentum magnitude (capped at 5%)
        strength = min(abs(self.momentum) / 0.05, 1.0)
        
        if self.momentum > 0 and self.acceleration >= 0:
            return strength, 'BULLISH'
        elif self.momentum < 0 and self.acceleration <= 0:
            return strength, 'BEARISH'
        else:
            # Momentum and acceleration disagree - potential reversal
            return strength * 0.5, 'REVERSAL'
        
    def compute_log_likelihood(self, volatilities, regime_model, regime):
        """
        Compute log-likelihood of observed volatilities under a specific regime.
        
        MLE Concept: The regime with highest likelihood best explains the data.
        log L(θ|x) = Σ log P(xᵢ|θ)
        
        Where θ represents the regime parameters (mean, std) and x is the volatility data.
        """
        if len(volatilities) == 0:
            return -np.inf
            
        mean = regime_model.emission_means[regime]
        std = regime_model.emission_stds[regime]
        
        # Sum of log-likelihoods (more numerically stable than product of likelihoods)
        log_likelihood = np.sum(stats.norm.logpdf(volatilities, mean, std))
        return log_likelihood
    
    def likelihood_ratio_test(self, volatilities, regime_model, regime_h0, regime_h1):
        """
        Perform Likelihood Ratio Test (LRT) for regime change detection.
        
        Hypothesis Test:
        H₀: Data comes from regime_h0 (null hypothesis - no regime change)
        H₁: Data comes from regime_h1 (alternative - regime has changed)
        
        Test Statistic: λ = -2 * [log L(H₀) - log L(H₁)]
        Under H₀, λ follows χ² distribution with df=1
        
        Returns: (test_statistic, p_value, reject_h0)
        """
        if len(volatilities) < 2:
            return 0, 1.0, False
            
        # Compute log-likelihoods under each hypothesis
        ll_h0 = self.compute_log_likelihood(volatilities, regime_model, regime_h0)
        ll_h1 = self.compute_log_likelihood(volatilities, regime_model, regime_h1)
        
        # Likelihood ratio test statistic
        # λ = -2 * (log L₀ - log L₁) = 2 * (log L₁ - log L₀)
        test_statistic = 2 * (ll_h1 - ll_h0)
        
        # Under H₀, test statistic follows chi-squared distribution
        # P-value: probability of observing this extreme a value if H₀ is true
        if test_statistic > 0:
            p_value = 1 - stats.chi2.cdf(test_statistic, df=1)
        else:
            p_value = 1.0  # Cannot reject H₀ if H₁ is not more likely
            
        # Reject H₀ if p-value < significance level
        reject_h0 = p_value < self.significance_level
        
        return test_statistic, p_value, reject_h0
    
    def compute_regime_transition_probability(self, regime_model, from_regime, to_regime):
        """
        Get the transition probability from Markov chain.
        
        P(Sₜ = j | Sₜ₋₁ = i) = T[i,j]
        
        Low transition probability makes a regime change more "surprising"
        and potentially more significant for trading.
        """
        return regime_model.transition_matrix[from_regime, to_regime]
    
    def generate_signal(self, bars, regime_model, current_price):
        """
        Generate trading signal using HMM inference with trend/momentum confirmation.
        
        Enhanced Signal Logic:
        1. Use MAP estimate for current regime (from posterior probabilities)
        2. Validate regime change using hypothesis testing
        3. Confirm with trend analysis (statistically significant trend)
        4. Check momentum alignment
        5. Consider mean reversion for extreme moves
        
        Trading Rules (with trend/momentum confirmation):
        - BUY when: volatility decreasing + (uptrend OR oversold z-score) + positive momentum
        - SELL when: volatility increasing + (downtrend OR overbought z-score) + negative momentum
        - HOLD during conflicting signals or uncertain periods
        """
        if len(bars) < self.min_bars_for_signal:
            return None
            
        current_bar = bars[-1]
        current_regime = current_bar.regime
        
        # Get MAP probability (posterior probability of current regime)
        map_probability = regime_model.state_probs[current_regime]
        
        # Track regime and price history
        self.regime_history.append(current_regime)
        self.volatility_history.append(current_bar.volatility)
        self.price_history.append(current_bar.close)
        
        # Keep only recent history
        window_size = max(self.trend_window, self.mean_reversion_window)
        if len(self.volatility_history) > window_size:
            self.volatility_history = self.volatility_history[-window_size:]
            self.regime_history = self.regime_history[-window_size:]
            self.price_history = self.price_history[-window_size:]
        
        # =====================================================================
        # COMPUTE TREND AND MOMENTUM INDICATORS
        # =====================================================================
        trend_slope, trend_r2, trend_pval = self.compute_trend(self.price_history)
        momentum, acceleration = self.compute_momentum(self.price_history)
        z_score = self.compute_z_score(self.price_history)
        
        trend_direction, trend_confidence = self.get_trend_direction()
        momentum_strength, momentum_dir = self.get_momentum_signal()
        
        # Check for regime transition
        if len(self.regime_history) < 2:
            return None
            
        previous_regime = self.regime_history[-2]
        
        # =====================================================================
        # HYPOTHESIS TEST: Is this regime change statistically significant?
        # =====================================================================
        recent_vols = np.array(self.volatility_history[-5:]) if len(self.volatility_history) >= 5 else np.array(self.volatility_history)
        test_stat, p_value, is_significant = self.likelihood_ratio_test(
            recent_vols, regime_model, previous_regime, current_regime
        )
        
        # Get transition probability
        trans_prob = self.compute_regime_transition_probability(
            regime_model, previous_regime, current_regime
        )
        
        # Create comprehensive signal info
        signal = {
            'timestamp': current_bar.timestamp,
            'price': current_price,
            'from_regime': previous_regime,
            'to_regime': current_regime,
            'map_probability': map_probability,
            'p_value': p_value,
            'is_significant': is_significant,
            'transition_prob': trans_prob,
            'test_statistic': test_stat,
            # Enhanced indicators
            'trend_direction': trend_direction,
            'trend_confidence': trend_confidence,
            'trend_slope': trend_slope,
            'trend_r_squared': trend_r2,
            'trend_p_value': trend_pval,
            'momentum': momentum,
            'momentum_direction': momentum_dir,
            'acceleration': acceleration,
            'z_score': z_score,
            'action': None,
            'executed': False
        }
        
        # =====================================================================
        # ENHANCED TRADING DECISION LOGIC
        # =====================================================================
        # We use a scoring system that combines:
        # 1. Regime signal (primary)
        # 2. Trend confirmation (secondary)
        # 3. Momentum alignment (secondary)
        # 4. Mean reversion opportunity (bonus)
        
        regime_names = ['LOW', 'MED', 'HIGH']
        buy_score = 0.0
        sell_score = 0.0
        reasons = []
        
        # --- REGIME COMPONENT (weight: 40%) ---
        regime_changed = (current_regime != previous_regime)
        if regime_changed and is_significant and map_probability >= self.confidence_threshold:
            if current_regime < previous_regime:  # Volatility decreasing
                buy_score += 0.4
                reasons.append(f"Regime↓({regime_names[previous_regime]}→{regime_names[current_regime]})")
            elif current_regime > previous_regime:  # Volatility increasing
                sell_score += 0.4
                reasons.append(f"Regime↑({regime_names[previous_regime]}→{regime_names[current_regime]})")
        
        # --- TREND COMPONENT (weight: 30%) ---
        # Only consider if trend is statistically significant
        if trend_pval < self.significance_level:
            if trend_direction == 'UP':
                buy_score += 0.3 * trend_confidence
                reasons.append(f"Trend↑(p={trend_pval:.3f})")
            elif trend_direction == 'DOWN':
                sell_score += 0.3 * trend_confidence
                reasons.append(f"Trend↓(p={trend_pval:.3f})")
        
        # --- MOMENTUM COMPONENT (weight: 20%) ---
        if momentum_strength > 0.3:  # Only consider meaningful momentum
            if momentum_dir == 'BULLISH':
                buy_score += 0.2 * momentum_strength
                reasons.append(f"Mom+({momentum*100:.1f}%)")
            elif momentum_dir == 'BEARISH':
                sell_score += 0.2 * momentum_strength
                reasons.append(f"Mom-({momentum*100:.1f}%)")
            elif momentum_dir == 'REVERSAL':
                # Potential reversal - reduce confidence
                buy_score *= 0.8
                sell_score *= 0.8
        
        # --- MEAN REVERSION COMPONENT (weight: 10%) ---
        # Z-score extreme values suggest mean reversion opportunity
        if abs(z_score) > self.mean_reversion_threshold:
            if z_score < -self.mean_reversion_threshold:  # Oversold
                buy_score += 0.1
                reasons.append(f"Oversold(z={z_score:.2f})")
            elif z_score > self.mean_reversion_threshold:  # Overbought
                sell_score += 0.1
                reasons.append(f"Overbought(z={z_score:.2f})")
        
        # =====================================================================
        # FINAL DECISION
        # =====================================================================
        # Require minimum combined score of 0.5 to act
        # This means at least regime signal + one confirmation
        min_score_threshold = 0.5
        
        # Also require regime signal to be present (don't trade on trend/momentum alone)
        regime_signal_present = regime_changed and is_significant
        
        if regime_signal_present:
            if buy_score >= min_score_threshold and buy_score > sell_score:
                if self.position <= 0:  # Not already long
                    signal['action'] = 'BUY'
                    signal['score'] = buy_score
                    signal['reason'] = ' + '.join(reasons)
            elif sell_score >= min_score_threshold and sell_score > buy_score:
                if self.position >= 0:  # Not already short
                    signal['action'] = 'SELL'
                    signal['score'] = sell_score
                    signal['reason'] = ' + '.join(reasons)
        
        # Store signal for analysis
        self.signals.append(signal)
        
        return signal if signal['action'] else None
    
    def execute_trade(self, signal, ib_app, contract, simulation_mode=True):
        """
        Execute trade via Interactive Brokers API.
        
        For paper trading, sends actual orders to IB TWS Paper account.
        """
        if not signal or not signal['action']:
            return False
            
        try:
            # Update position tracking
            if signal['action'] == 'BUY':
                if self.position == -1:  # Close short first
                    self._record_trade('CLOSE_SHORT', signal['price'])
                self.position = 1
                self.position_price = signal['price']
                
            elif signal['action'] == 'SELL':
                if self.position == 1:  # Close long first
                    self._record_trade('CLOSE_LONG', signal['price'])
                self.position = -1
                self.position_price = signal['price']
            
            # In simulation mode, just track the trade
            if simulation_mode:
                signal['executed'] = True
                print(f"\n{'='*70}")
                print(f"TRADE SIGNAL EXECUTED (SIMULATION)")
                print(f"{'='*70}")
                print(f"Action: {signal['action']}")
                print(f"Price: ${signal['price']:.2f}")
                print(f"Score: {signal.get('score', 0):.2f}")
                print(f"Reason: {signal.get('reason', 'N/A')}")
                print(f"\n--- Regime Analysis ---")
                print(f"MAP Probability: {signal['map_probability']*100:.1f}%")
                print(f"P-Value (LRT): {signal['p_value']:.4f}")
                print(f"Statistically Significant: {signal['is_significant']}")
                print(f"\n--- Trend & Momentum ---")
                print(f"Trend: {signal.get('trend_direction', 'N/A')} (p={signal.get('trend_p_value', 1):.4f}, R²={signal.get('trend_r_squared', 0):.3f})")
                print(f"Momentum: {signal.get('momentum', 0)*100:+.2f}% ({signal.get('momentum_direction', 'N/A')})")
                print(f"Z-Score: {signal.get('z_score', 0):.2f}σ")
                print(f"{'='*70}\n")
                return True
            
            # Live trading: Send order to IB
            if not hasattr(ib_app, 'next_order_id'):
                print("Error: No valid order ID available")
                return False
                
            order = Order()
            order.action = signal['action']
            order.totalQuantity = self.position_size
            order.orderType = "MKT"  # Market order for immediate execution
            order.eTradeOnly = False
            order.firmQuoteOnly = False
            
            order_id = ib_app.next_order_id
            ib_app.next_order_id += 1
            
            ib_app.placeOrder(order_id, contract, order)
            signal['executed'] = True
            signal['order_id'] = order_id
            
            print(f"\n{'='*60}")
            print(f"ORDER PLACED - ID: {order_id}")
            print(f"{'='*60}")
            print(f"Action: {signal['action']} {self.position_size} shares")
            print(f"Type: MARKET ORDER")
            print(f"Reason: {signal.get('reason', 'N/A')}")
            print(f"MAP Probability: {signal['map_probability']*100:.1f}%")
            print(f"P-Value: {signal['p_value']:.4f}")
            print(f"{'='*60}\n")
            
            return True
            
        except Exception as e:
            print(f"Trade execution error: {e}")
            return False
    
    def _record_trade(self, action, price):
        """Record a completed trade for P&L tracking."""
        if action == 'CLOSE_LONG' and self.position == 1:
            pnl = (price - self.position_price) * self.position_size
            self.realized_pnl += pnl
            self.trades.append({
                'type': 'LONG',
                'entry': self.position_price,
                'exit': price,
                'pnl': pnl,
                'timestamp': datetime.now()
            })
        elif action == 'CLOSE_SHORT' and self.position == -1:
            pnl = (self.position_price - price) * self.position_size
            self.realized_pnl += pnl
            self.trades.append({
                'type': 'SHORT',
                'entry': self.position_price,
                'exit': price,
                'pnl': pnl,
                'timestamp': datetime.now()
            })
    
    def update_unrealized_pnl(self, current_price):
        """Calculate unrealized P&L for open position."""
        if self.position == 1:
            self.unrealized_pnl = (current_price - self.position_price) * self.position_size
        elif self.position == -1:
            self.unrealized_pnl = (self.position_price - current_price) * self.position_size
        else:
            self.unrealized_pnl = 0.0
        return self.unrealized_pnl
    
    def get_statistics(self):
        """Return comprehensive strategy performance statistics."""
        stats = {
            'total_trades': len(self.trades),
            'winning_trades': 0,
            'losing_trades': 0,
            'win_rate': 0.0,
            'total_pnl': self.realized_pnl,
            'avg_pnl': 0.0,
            'max_win': 0.0,
            'max_loss': 0.0,
            # Current indicator values
            'current_trend': self.get_trend_direction()[0],
            'trend_confidence': self.get_trend_direction()[1],
            'trend_slope': self.trend_slope,
            'trend_r_squared': self.trend_r_squared,
            'momentum': self.momentum,
            'acceleration': self.acceleration,
            'z_score': self.z_score,
            # Signal analysis
            'total_signals': len(self.signals),
            'executed_signals': len([s for s in self.signals if s.get('executed', False)])
        }
        
        if self.trades:
            winning = [t for t in self.trades if t['pnl'] > 0]
            losing = [t for t in self.trades if t['pnl'] <= 0]
            
            stats['winning_trades'] = len(winning)
            stats['losing_trades'] = len(losing)
            stats['win_rate'] = len(winning) / len(self.trades) * 100
            stats['avg_pnl'] = self.realized_pnl / len(self.trades)
            
            if winning:
                stats['max_win'] = max(t['pnl'] for t in winning)
            if losing:
                stats['max_loss'] = min(t['pnl'] for t in losing)
        
        return stats
    
    def get_indicator_summary(self):
        """
        Get a formatted summary of current technical indicators.
        Useful for display in the UI.
        """
        trend_dir, trend_conf = self.get_trend_direction()
        mom_strength, mom_dir = self.get_momentum_signal()
        
        return {
            'trend': f"{trend_dir} ({trend_conf*100:.0f}%)" if trend_dir != 'NEUTRAL' else "NEUTRAL",
            'trend_significant': self.trend_p_value < self.significance_level,
            'momentum': f"{self.momentum*100:+.2f}%",
            'momentum_direction': mom_dir,
            'z_score': f"{self.z_score:.2f}σ",
            'mean_reversion_signal': abs(self.z_score) > self.mean_reversion_threshold
        }


# =============================================================================
# MAIN DASHBOARD APPLICATION
# =============================================================================
class LiveMarketDashboard:
    """Main application - live market data dashboard with rolling OHLC chart and regime detection."""

    def __init__(self, root):
        self.root = root
        self.root.title("Live Market Data Dashboard")
        self.root.geometry("1200x800")
        self.root.configure(bg='#0d1117')               # Dark background

        # Initialize dark theme styling
        self.style = ttk.Style()
        self.style.theme_use('clam')
        self.configure_dark_theme()

        # Create IB API instance with callback for tick data
        self.ib_app = IBApp(callback=self.on_tick_data)
        self.connected = False                          # TWS connection status
        self.streaming = False                          # Market data streaming status

        # Data storage configuration
        self.bar_duration = 5                           # Seconds per OHLC bar
        self.max_bars = 10                              # Maximum bars to display (rolling window)
        self.ohlc_bars = deque(maxlen=self.max_bars)   # Completed bars
        self.current_bar = None                         # Bar currently being built
        self.bar_start_time = None                      # When current bar started
        self.price_history = deque(maxlen=100)         # Recent tick prices
        self.last_update_time = None
        self.regime_model = MarkovRegime()             # Volatility regime classifier

        # Thread synchronization
        self.bar_lock = threading.Lock()               # Protect bar data from race conditions
        self.update_thread = None                       # Background thread for bar management
        self.running = False                            # Control flag for loops
        self.tick_queue = queue.Queue()                 # Thread-safe queue for tick data
        self.pending_signal = None                      # Signal waiting to be processed on main thread
        self.pending_price = None                       # Price for pending signal
        # Add after self.streaming = False
        self.simulation_mode = False
        self.simulation_data = []
        self.simulation_index = 0
        self.playback_speed = 1.0
        
        # Trading strategy
        self.trading_strategy = TradingStrategy()
        self.trading_enabled = False                    # Toggle for auto-trading
        self.current_contract = None                    # Current trading contract
        
        # Build UI and chart
        self.setup_ui()
        self.setup_chart()

    def configure_dark_theme(self):
        # Apply GitHub-inspired dark theme to ttk widgets
        bg_color = '#0d1117'                            # Main background
        fg_color = '#c9d1d9'                            # Text color
        accent_color = '#238636'                        # Button accent (green)
        entry_bg = '#161b22'                            # Input field background

        # Configure each widget type
        self.style.configure('TFrame', background=bg_color)
        self.style.configure('TLabelframe', background=bg_color, foreground=fg_color)
        self.style.configure('TLabelframe.Label', background=bg_color, foreground=fg_color,
                            font=('Segoe UI', 10, 'bold'))
        self.style.configure('TLabel', background=bg_color, foreground=fg_color,
                            font=('Segoe UI', 10))
        self.style.configure('TButton', background=accent_color, foreground='white',
                            font=('Segoe UI', 9, 'bold'), padding=(10, 5))
        self.style.map('TButton',
                      background=[('active', '#2ea043'), ('disabled', '#21262d')])
        self.style.configure('TEntry', fieldbackground=entry_bg, foreground=fg_color,
                            insertcolor=fg_color)
        # Red accent style for stop/disconnect buttons
        self.style.configure('Accent.TButton', background='#da3633', foreground='white')
        self.style.map('Accent.TButton',
                      background=[('active', '#f85149'), ('disabled', '#21262d')])
        # Add after the recal_btn
        
    def update_speed(self, val):
        self.playback_speed = float(val)
        self.speed_label.config(text=f"{self.playback_speed:.1f}x")
    def setup_ui(self):
        # Build the user interface layout
        main_frame = ttk.Frame(self.root, padding="15")
        main_frame.grid(row=0, column=0, sticky='nsew')

        # Configure grid weights for responsive resizing
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=1)
        main_frame.rowconfigure(2, weight=1)            # Chart row expands

        # ----- HEADER SECTION -----
        header_frame = ttk.Frame(main_frame)
        header_frame.grid(row=0, column=0, sticky='ew', pady=(0, 15))

        # Application title
        title_label = tk.Label(header_frame, text="◈ LIVE REGIME SWITCHING", 
                              font=('JetBrains Mono', 18, 'bold'),
                              bg='#0d1117', fg='#58a6ff')
        title_label.pack(side='left')

        # Connection status indicator (red=disconnected, green=connected, blue=streaming)
        self.status_indicator = tk.Label(header_frame, text="● DISCONNECTED",
                                        font=('Segoe UI', 10, 'bold'),
                                        bg='#0d1117', fg='#f85149')
        self.status_indicator.pack(side='right', padx=10)

        # ----- CONTROL PANEL -----
        control_frame = ttk.LabelFrame(main_frame, text="Control Panel", padding="10")
        control_frame.grid(row=1, column=0, sticky='ew', pady=(0, 15))

        # Connection inputs (host and port)
        conn_section = ttk.Frame(control_frame)
        conn_section.pack(fill='x', pady=(0, 10))

        ttk.Label(conn_section, text="Host:").pack(side='left', padx=(0, 5))
        self.host_var = tk.StringVar(value="127.0.0.1") # TWS default host
        host_entry = ttk.Entry(conn_section, textvariable=self.host_var, width=12)
        host_entry.pack(side='left', padx=(0, 15))

        ttk.Label(conn_section, text="Port:").pack(side='left', padx=(0, 5))
        self.port_var = tk.StringVar(value="7497")      # 7497=paper, 7496=live
        port_entry = ttk.Entry(conn_section, textvariable=self.port_var, width=8)
        port_entry.pack(side='left', padx=(0, 15))

        # Connect/Disconnect buttons
        self.connect_btn = ttk.Button(conn_section, text="Connect", command=self.connect_ib)
        self.connect_btn.pack(side='left', padx=(0, 5))

        self.disconnect_btn = ttk.Button(conn_section, text="Disconnect", 
                                         command=self.disconnect_ib, state='disabled',
                                         style='Accent.TButton')
        self.disconnect_btn.pack(side='left')

        # Visual separator between connection and data sections
        sep = ttk.Separator(control_frame, orient='horizontal')
        sep.pack(fill='x', pady=10)

        # ----- DATA STREAMING SECTION -----
        data_section = ttk.Frame(control_frame)
        data_section.pack(fill='x')

        # Symbol input field
        ttk.Label(data_section, text="Symbol:").pack(side='left', padx=(0, 5))
        self.symbol_var = tk.StringVar(value="AAPL")
        symbol_entry = ttk.Entry(data_section, textvariable=self.symbol_var, width=10,
                                font=('JetBrains Mono', 11))
        symbol_entry.pack(side='left', padx=(0, 15))

        # Start/Stop stream button
        self.stream_btn = ttk.Button(data_section, text="▶ Start Stream", 
                                     command=self.toggle_stream, state='disabled')
        self.stream_btn.pack(side='left', padx=(0, 5))

        # Recalibrate regime model button
        self.recal_btn = ttk.Button(data_section, text="⟳ Recalibrate", 
                                    command=self.recalibrate_model, state='disabled')
        self.recal_btn.pack(side='left', padx=(0, 15))

        # Speed control for playback
        ttk.Label(data_section, text="Speed:").pack(side='left', padx=(15, 5))
        self.speed_var = tk.DoubleVar(value=1.0)
        speed_scale = ttk.Scale(data_section, from_=0.5, to=5.0, 
                            variable=self.speed_var, orient='horizontal',
                            length=100, command=self.update_speed)
        speed_scale.pack(side='left', padx=(0, 5))
        self.speed_label = tk.Label(data_section, text="1.0x",
                                font=('JetBrains Mono', 9, 'bold'),
                                bg='#0d1117', fg='#8b949e')
        self.speed_label.pack(side='left')

        # Live price display (right-aligned)
        price_frame = ttk.Frame(data_section)
        price_frame.pack(side='right')

        ttk.Label(price_frame, text="Last Price:", 
                 font=('Segoe UI', 10)).pack(side='left', padx=(0, 5))
        self.price_label = tk.Label(price_frame, text="---.--",
                                   font=('JetBrains Mono', 16, 'bold'),
                                   bg='#0d1117', fg='#7ee787')
        self.price_label.pack(side='left')

        # ----- CHART FRAME -----
        chart_frame = ttk.LabelFrame(main_frame, text="Live OHLC with Markov Regime (5s Bars)", padding="10")
        chart_frame.grid(row=2, column=0, sticky='nsew')
        chart_frame.columnconfigure(0, weight=1)
        chart_frame.rowconfigure(0, weight=1)

        self.chart_container = ttk.Frame(chart_frame)
        self.chart_container.grid(row=0, column=0, sticky='nsew')
        self.chart_container.columnconfigure(0, weight=1)
        self.chart_container.rowconfigure(0, weight=1)

        # ----- STATISTICS BAR -----
        stats_frame = ttk.Frame(main_frame)
        stats_frame.grid(row=3, column=0, sticky='ew', pady=(10, 0))

        # Create stat labels: Bars count, High, Low, Current Regime, Ticks per bar
        self.stats_labels = {}
        stats = [('Bars', '0'), ('High', '--'), ('Low', '--'), ('Regime', '--'), 
         ('Ticks/Bar', '0'), ('Prob', '--')]
        
        for i, (name, val) in enumerate(stats):
            frame = ttk.Frame(stats_frame)
            frame.pack(side='left', padx=15)
            ttk.Label(frame, text=f"{name}:", font=('Segoe UI', 9)).pack(side='left')
            label = tk.Label(frame, text=val, font=('JetBrains Mono', 10, 'bold'),
                           bg='#0d1117', fg='#8b949e')
            label.pack(side='left', padx=(5, 0))
            self.stats_labels[name] = label

        # ----- TRADING PANEL -----
        trading_frame = ttk.LabelFrame(main_frame, text="HMM Trading Strategy", padding="10")
        trading_frame.grid(row=4, column=0, sticky='ew', pady=(10, 0))
        
        # Trading controls (left side)
        trade_controls = ttk.Frame(trading_frame)
        trade_controls.pack(side='left', fill='x')
        
        # Auto-trade toggle button
        self.trade_enabled_var = tk.BooleanVar(value=False)
        self.trade_btn = ttk.Button(trade_controls, text="Enable Auto-Trade", 
                                   command=self.toggle_trading)
        self.trade_btn.pack(side='left', padx=(0, 10))
        
        # Confidence threshold
        ttk.Label(trade_controls, text="Confidence:").pack(side='left', padx=(10, 5))
        self.confidence_var = tk.DoubleVar(value=0.70)
        confidence_scale = ttk.Scale(trade_controls, from_=0.5, to=0.95, 
                                    variable=self.confidence_var, orient='horizontal',
                                    length=80, command=self.update_confidence)
        confidence_scale.pack(side='left', padx=(0, 5))
        self.confidence_label = tk.Label(trade_controls, text="70%",
                                        font=('JetBrains Mono', 9, 'bold'),
                                        bg='#0d1117', fg='#8b949e')
        self.confidence_label.pack(side='left')
        
        # Significance level (alpha)
        ttk.Label(trade_controls, text="α:").pack(side='left', padx=(15, 5))
        self.alpha_var = tk.DoubleVar(value=0.05)
        alpha_scale = ttk.Scale(trade_controls, from_=0.01, to=0.10, 
                               variable=self.alpha_var, orient='horizontal',
                               length=60, command=self.update_alpha)
        alpha_scale.pack(side='left', padx=(0, 5))
        self.alpha_label = tk.Label(trade_controls, text="0.05",
                                   font=('JetBrains Mono', 9, 'bold'),
                                   bg='#0d1117', fg='#8b949e')
        self.alpha_label.pack(side='left')
        
        # Trading status labels (right side)
        trade_status = ttk.Frame(trading_frame)
        trade_status.pack(side='right', fill='x')
        
        # Position display
        pos_frame = ttk.Frame(trade_status)
        pos_frame.pack(side='left', padx=15)
        ttk.Label(pos_frame, text="Position:", font=('Segoe UI', 9)).pack(side='left')
        self.position_label = tk.Label(pos_frame, text="FLAT",
                                       font=('JetBrains Mono', 10, 'bold'),
                                       bg='#0d1117', fg='#8b949e')
        self.position_label.pack(side='left', padx=(5, 0))
        
        # P&L display
        pnl_frame = ttk.Frame(trade_status)
        pnl_frame.pack(side='left', padx=15)
        ttk.Label(pnl_frame, text="P&L:", font=('Segoe UI', 9)).pack(side='left')
        self.pnl_label = tk.Label(pnl_frame, text="$0.00",
                                 font=('JetBrains Mono', 10, 'bold'),
                                 bg='#0d1117', fg='#8b949e')
        self.pnl_label.pack(side='left', padx=(5, 0))
        
        # Last signal display
        signal_frame = ttk.Frame(trade_status)
        signal_frame.pack(side='left', padx=15)
        ttk.Label(signal_frame, text="Signal:", font=('Segoe UI', 9)).pack(side='left')
        self.signal_label = tk.Label(signal_frame, text="--",
                                    font=('JetBrains Mono', 10, 'bold'),
                                    bg='#0d1117', fg='#8b949e')
        self.signal_label.pack(side='left', padx=(5, 0))
        
        # P-value display
        pval_frame = ttk.Frame(trade_status)
        pval_frame.pack(side='left', padx=15)
        ttk.Label(pval_frame, text="p-value:", font=('Segoe UI', 9)).pack(side='left')
        self.pvalue_label = tk.Label(pval_frame, text="--",
                                    font=('JetBrains Mono', 10, 'bold'),
                                    bg='#0d1117', fg='#8b949e')
        self.pvalue_label.pack(side='left', padx=(5, 0))

    def toggle_trading(self):
        """Toggle auto-trading on/off."""
        self.trading_enabled = not self.trading_enabled
        if self.trading_enabled:
            self.trade_btn.config(text="Disable Auto-Trade", style='Accent.TButton')
            print("\n[TRADING] Auto-trading ENABLED")
            print(f"[TRADING] Confidence threshold: {self.trading_strategy.confidence_threshold*100:.0f}%")
            print(f"[TRADING] Significance level (α): {self.trading_strategy.significance_level}")
        else:
            self.trade_btn.config(text="Enable Auto-Trade", style='TButton')
            print("\n[TRADING] Auto-trading DISABLED")
    
    def update_confidence(self, val):
        """Update confidence threshold for trading signals."""
        confidence = float(val)
        self.trading_strategy.confidence_threshold = confidence
        self.confidence_label.config(text=f"{confidence*100:.0f}%")
    
    def update_alpha(self, val):
        """Update significance level (alpha) for hypothesis testing."""
        alpha = float(val)
        self.trading_strategy.significance_level = alpha
        self.alpha_label.config(text=f"{alpha:.2f}")

    def setup_chart(self):
        # Initialize matplotlib figure with dark theme
        plt.style.use('dark_background')
        
        # Create figure and axis
        self.fig, self.ax = plt.subplots(figsize=(12, 6), facecolor='#0d1117')
        self.ax.set_facecolor('#161b22')
        
        # Style axis spines (borders) and ticks
        self.ax.tick_params(colors='#8b949e', labelsize=9)
        self.ax.spines['bottom'].set_color('#30363d')
        self.ax.spines['top'].set_color('#30363d')
        self.ax.spines['left'].set_color('#30363d')
        self.ax.spines['right'].set_color('#30363d')
        self.ax.grid(True, alpha=0.2, color='#30363d', linestyle='--')
        
        # Set axis labels and initial title
        self.ax.set_xlabel('Time', color='#8b949e', fontsize=10)
        self.ax.set_ylabel('Price', color='#8b949e', fontsize=10)
        self.ax.set_title('Waiting for data...', color='#c9d1d9', fontsize=12, fontweight='bold')

        # Embed matplotlib canvas in tkinter
        self.canvas = FigureCanvasTkAgg(self.fig, self.chart_container)
        self.canvas.get_tk_widget().grid(row=0, column=0, sticky='nsew')

        self.fig.tight_layout()
        self.canvas.draw()

    def create_contract(self, symbol):
        # Create IB Contract object for US stock
        contract = Contract()
        contract.symbol = symbol.upper()
        contract.secType = "STK"                        # Stock
        contract.exchange = "SMART"                     # IB smart routing
        contract.primaryExchange = "NASDAQ"             # Primary exchange for AAPL
        contract.currency = "USD"
        return contract

    def connect_ib(self):
        # Establish connection to TWS in background thread
        try:
            host = self.host_var.get()
            port = int(self.port_var.get())

            def connect_thread():
                # Run IB client message loop (blocking)
                try:
                    self.ib_app.connect(host, port, clientId=1)
                    self.ib_app.run()
                except Exception as e:
                    print(f"Connection error: {e}")

            # Start connection in daemon thread (dies with main program)
            thread = threading.Thread(target=connect_thread, daemon=True)
            thread.start()

            # Wait up to 5 seconds for connection confirmation
            for _ in range(50):
                if self.ib_app.connected:
                    break
                time.sleep(0.1)

            # Update UI based on connection result
            if self.ib_app.connected:
                self.connected = True
                self.connect_btn.config(state='disabled')
                self.disconnect_btn.config(state='normal')
                self.stream_btn.config(state='normal')
                self.status_indicator.config(text="● CONNECTED", fg='#7ee787')
            else:
                messagebox.showerror("Error", "Failed to connect to TWS")

        except Exception as e:
            messagebox.showerror("Error", f"Connection error: {e}")

    def disconnect_ib(self):
        # Clean disconnect from TWS
        try:
            if self.streaming:
                self.stop_stream()
            
            self.ib_app.disconnect()
            self.connected = False
            # Reset UI to disconnected state
            self.connect_btn.config(state='normal')
            self.disconnect_btn.config(state='disabled')
            self.stream_btn.config(state='disabled')
            self.status_indicator.config(text="● DISCONNECTED", fg='#f85149')

        except Exception as e:
            print(f"Disconnect error: {e}")

    def toggle_stream(self):
        # Toggle between starting and stopping market data stream
        if not self.streaming:
            self.start_stream()
        else:
            self.stop_stream()

    def start_stream(self):
        """Start simulation mode using historical data replay."""
        if not self.connected:
            return

        symbol = self.symbol_var.get().upper()
        if not symbol:
            messagebox.showerror("Error", "Please enter a symbol")
            return

        # Reset state
        with self.bar_lock:
            self.ohlc_bars.clear()
            self.current_bar = None
            self.bar_start_time = None
            self.price_history.clear()
            self.regime_model = MarkovRegime()
            
        # Reset trading strategy for new session
        self.trading_strategy = TradingStrategy()
        self.trading_strategy.confidence_threshold = self.confidence_var.get()
        self.trading_strategy.significance_level = self.alpha_var.get()

        contract = self.create_contract(symbol)
        self.current_contract = contract  # Store for trading
        
        # Fetch historical data (2 hours of 1-minute bars = 120 bars)
        self.status_indicator.config(text="● LOADING DATA...", fg='#d29922')
        self.ib_app.historical_data.clear()
        self.ib_app.hist_done.clear()
        # Use empty string for end date (means "now") - more reliable than explicit datetime
        self.ib_app.reqHistoricalData(2, contract, "", "7200 S", "1 min", "TRADES", 0, 1, False, [])
        
        if not self.ib_app.hist_done.wait(timeout=15):
            messagebox.showerror("Error", "Timeout fetching historical data")
            self.status_indicator.config(text="● CONNECTED", fg='#7ee787')
            return
            
        if 2 not in self.ib_app.historical_data or len(self.ib_app.historical_data[2]) == 0:
            messagebox.showerror("Error", "No historical data received")
            self.status_indicator.config(text="● CONNECTED", fg='#7ee787')
            return

        hist_data = self.ib_app.historical_data[2]
        print(f"\nReceived {len(hist_data)} historical bars for {symbol}")

        # Split data: 50% for calibration, 50% for simulation
        split_point = len(hist_data) // 2
        calibration_bars = hist_data[:split_point]
        simulation_bars = hist_data[split_point:]

        # Calibrate regime model
        print(f"Calibrating with {len(calibration_bars)} bars...")
        self.regime_model.calibrate(calibration_bars)

        # Setup simulation
        self.simulation_data = simulation_bars
        self.simulation_index = 0
        self.simulation_mode = True

        print(f"Starting simulation with {len(self.simulation_data)} bars")
        print(f"Playback speed: {self.playback_speed}x")
        print(f"Total simulation time: ~{len(self.simulation_data) * 5 / self.playback_speed:.0f} seconds\n")

        # Start simulation
        self.streaming = True
        self.running = True
        self.stream_btn.config(text="■ Stop Simulation", style='Accent.TButton')
        self.recal_btn.config(state='normal')
        self.status_indicator.config(text=f"● SIMULATING {symbol}", fg='#d29922')

        # Clear tick queue before starting
        while not self.tick_queue.empty():
            try:
                self.tick_queue.get_nowait()
            except queue.Empty:
                break
        
        # Launch simulation thread
        self.update_thread = threading.Thread(target=self.simulation_loop, daemon=True)
        self.update_thread.start()

        # Start tick queue processing on main thread
        self.root.after(50, self.process_tick_queue)
        
        # Start chart updates
        self.update_chart_loop()

    def stop_stream(self):
        # Stop market data streaming and cleanup
        self.running = False
        self.streaming = False
        
        # Clear tick queue
        while not self.tick_queue.empty():
            try:
                self.tick_queue.get_nowait()
            except queue.Empty:
                break
        
        # Cancel market data subscription
        try:
            self.ib_app.cancelMktData(1)
        except:
            pass

        # Reset UI to connected (not streaming) state
        try:
            self.stream_btn.config(text="▶ Start Stream", style='TButton')
            self.recal_btn.config(state='disabled')
            self.status_indicator.config(text="● CONNECTED", fg='#7ee787')
        except Exception as e:
            print(f"Error updating UI in stop_stream: {e}")

    def recalibrate_model(self):
        # Fetch fresh historical data and recalibrate regime thresholds
        if not self.streaming:
            return
        contract = self.create_contract(self.symbol_var.get().upper())
        self.ib_app.historical_data.clear()
        self.ib_app.hist_done.clear()
        # Request latest 1 hour of 1-minute bars for recalibration
        self.ib_app.reqHistoricalData(3, contract, "", "3600 S", "1 min", "TRADES", 0, 1, False, [])
        if self.ib_app.hist_done.wait(timeout=10) and 3 in self.ib_app.historical_data:
            with self.bar_lock:
                self.regime_model.calibrate(self.ib_app.historical_data[3])
            print(f"Recalibrated with {len(self.ib_app.historical_data[3])} bars")

    def on_tick_data(self, data_type, value, timestamp):
        # Callback invoked by IBApp when new tick arrives (from IB API thread)
        if data_type == 'price' and value > 0:
            # Put in queue for main thread to process
            self.tick_queue.put(('price', value, timestamp))
    
    def process_tick_queue(self):
        """Process tick data from queue on main thread - thread safe."""
        if not self.running:
            return
        
        try:
            # Process all available ticks in queue
            processed = 0
            while not self.tick_queue.empty() and processed < 50:  # Limit per cycle
                try:
                    data_type, value, timestamp = self.tick_queue.get_nowait()
                    
                    if data_type == 'price' and value is not None and value > 0:
                        with self.bar_lock:
                            # Store tick in price history
                            self.price_history.append((timestamp, value))
                            
                            # Create new bar or update existing one
                            if self.current_bar is None:
                                self.current_bar = OHLCBar(timestamp, value)
                                self.bar_start_time = timestamp
                            else:
                                self.current_bar.update(value)
                        
                        # Update price display (already on main thread)
                        self.price_label.config(text=f"{value:.2f}")
                    
                    elif data_type == 'bar_complete':
                        # Finalize the bar
                        self._finalize_current_bar()
                    
                    elif data_type == 'simulation_complete':
                        bars_count = value
                        messagebox.showinfo("Complete", 
                            f"Simulation finished: {bars_count} bars replayed")
                        self.stop_stream()
                        return
                    
                    elif data_type == 'error':
                        print(f"Simulation error: {value}")
                        self.stop_stream()
                        return
                    
                    processed += 1
                    
                except queue.Empty:
                    break
                    
        except Exception as e:
            print(f"Error processing tick queue: {e}")
        
        # Schedule next queue processing
        if self.running:
            self.root.after(50, self.process_tick_queue)
    
    def _finalize_current_bar(self):
        """Finalize the current bar and process trading signals - runs on main thread."""
        with self.bar_lock:
            if self.current_bar is not None:
                # Determine regime for completed bar
                all_bars = list(self.ohlc_bars) + [self.current_bar]
                regime = self.regime_model.get_regime(all_bars)
                self.current_bar.regime = regime
                # Add to completed bars
                self.ohlc_bars.append(self.current_bar)
                
                # Generate trading signal using HMM-based strategy
                current_price = self.current_bar.close
                signal = self.trading_strategy.generate_signal(
                    all_bars, self.regime_model, current_price
                )
                
                # Execute trade if auto-trading is enabled and signal is valid
                if signal and self.trading_enabled:
                    self.trading_strategy.execute_trade(
                        signal, self.ib_app, self.current_contract, 
                        simulation_mode=self.simulation_mode
                    )
                
                # Reset for next bar
                self.current_bar = None
                self.bar_start_time = None
                
                # Update trading UI (already on main thread)
                self.update_trading_ui(signal, current_price)

    def simulation_loop(self):
        """Replay historical data as if it were live - simulates tick-by-tick."""
        base_delay = 5.0  # 5 seconds per bar in real-time
        
        try:
            while self.running and self.simulation_index < len(self.simulation_data):
                bar_data = self.simulation_data[self.simulation_index]
                
                # Generate realistic tick sequence within each bar
                # Start with open, include high/low, end with close
                prices = [bar_data['o']]
                
                # Add 8 intermediate ticks between low and high
                for _ in range(8):
                    price = bar_data['l'] + np.random.random() * (bar_data['h'] - bar_data['l'])
                    prices.append(price)
                
                # Ensure high and low are visited
                prices.append(bar_data['h'])
                prices.append(bar_data['l'])
                prices.append(bar_data['c'])
                
                # Shuffle middle ticks for realism (keep open first, close last)
                middle = list(prices[1:-1])
                np.random.shuffle(middle)
                prices = [prices[0]] + middle + [prices[-1]]
                
                # Send ticks with speed-adjusted delay
                tick_delay = (base_delay / len(prices)) / self.playback_speed
                
                for price in prices:
                    if not self.running:
                        break
                    # Put tick data in queue instead of direct call
                    self.tick_queue.put(('price', price, datetime.now()))
                    time.sleep(tick_delay)
                
                # Signal bar completion via queue
                self.tick_queue.put(('bar_complete', None, None))
                
                self.simulation_index += 1
            
            # Simulation complete - signal via queue
            if self.running:
                self.tick_queue.put(('simulation_complete', self.simulation_index, None))
                
        except Exception as e:
            print(f"Simulation loop error: {e}")
            self.tick_queue.put(('error', str(e), None))

    def update_chart_loop(self):
        # Periodic chart refresh loop (runs on main thread via after())
        if not self.running:
            return
        try:
            self.draw_ohlc_chart()
            self.update_stats()
        except Exception as e:
            print(f"Chart update error: {e}")
        # Schedule next update in 200ms, save ID for cleanup on close
        self._after_id = self.root.after(200, self.update_chart_loop)

    def draw_ohlc_chart(self):
        # Render the candlestick chart with regime background colors
        try:
            self.ax.clear()
        except Exception as e:
            print(f"Error clearing axis: {e}")
            return
        
        # Get thread-safe copy of bar data
        with self.bar_lock:
            bars = list(self.ohlc_bars)
            current = self.current_bar
            # Make a deep copy of current bar to avoid race conditions
            if current is not None:
                current_copy = OHLCBar(current.timestamp, current.open)
                current_copy.high = current.high
                current_copy.low = current.low
                current_copy.close = current.close
                current_copy.tick_count = current.tick_count
                current_copy.regime = current.regime
                current = current_copy

        # Include the currently forming bar in display
        if current is not None:
            bars = bars + [current]

        # Show placeholder if no data yet
        if not bars:
            self.ax.set_facecolor('#161b22')
            self.ax.set_title('Waiting for data...', color='#c9d1d9', fontsize=12, fontweight='bold')
            self.ax.grid(True, alpha=0.2, color='#30363d', linestyle='--')
            self.canvas.draw_idle()
            return

        # Calculate price range for y-axis with padding
        all_prices = [bar.low for bar in bars] + [bar.high for bar in bars]
        price_min, price_max = min(all_prices), max(all_prices)
        price_range = price_max - price_min
        padding = max(price_range * 0.1, 0.01)
        y_min, y_max = price_min - padding, price_max + padding

        # Assign current regime to the forming bar (preview - actual regime determined when bar completes)
        # Note: Don't call get_regime here as it would corrupt the Markov state
        if current is not None:
            current.regime = self.regime_model.current_state

        width = 0.6                                     # Candlestick body width
        for i, bar in enumerate(bars):
            # Draw regime background rectangle spanning full height
            bg = Rectangle((i - 0.5, y_min), 1, y_max - y_min, 
                          facecolor=self.regime_model.bg_colors[bar.regime], alpha=0.4, zorder=0)
            self.ax.add_patch(bg)

            # Determine candlestick color: green=bullish (close>=open), red=bearish
            color, edge_color = ('#3fb950', '#7ee787') if bar.close >= bar.open else ('#f85149', '#ff7b72')
            body_bottom, body_height = min(bar.open, bar.close), max(abs(bar.close - bar.open), 0.001)

            # Draw candlestick body
            rect = Rectangle((i - width/2, body_bottom), width, body_height, 
                            facecolor=color, edgecolor=edge_color, linewidth=1.5, alpha=0.9, zorder=2)
            self.ax.add_patch(rect)
            
            # Draw lower wick (low to body bottom)
            self.ax.plot([i, i], [bar.low, body_bottom], color=edge_color, linewidth=1.5, zorder=1)
            # Draw upper wick (body top to high)
            self.ax.plot([i, i], [body_bottom + body_height, bar.high], color=edge_color, linewidth=1.5, zorder=1)

            # Highlight current forming bar with vertical dotted line
            if i == len(bars) - 1 and current is not None:
                self.ax.axvline(x=i, color='#58a6ff', alpha=0.3, linestyle=':', linewidth=2)

        # Configure axes appearance
        self.ax.set_facecolor('#161b22')
        
        # Set x-axis labels to bar timestamps
        x_labels = [bar.timestamp.strftime('%H:%M:%S') for bar in bars]
        self.ax.set_xticks(range(len(bars)))
        self.ax.set_xticklabels(x_labels, rotation=45, ha='right', fontsize=8)
        self.ax.set_ylim(y_min, y_max)
        self.ax.set_xlim(-0.5, max(self.max_bars - 0.5, len(bars) - 0.5))
        # Format y-axis to show 3 decimal places (not scientific notation)
        self.ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.3f}'))

        # Apply axis styling
        self.ax.tick_params(colors='#8b949e', labelsize=9)
        self.ax.spines['bottom'].set_color('#30363d')
        self.ax.spines['top'].set_color('#30363d')
        self.ax.spines['left'].set_color('#30363d')
        self.ax.spines['right'].set_color('#30363d')
        self.ax.grid(True, alpha=0.2, color='#30363d', linestyle='--')

        self.ax.set_xlabel('Time', color='#8b949e', fontsize=10)
        self.ax.set_ylabel('Price', color='#8b949e', fontsize=10)
        
        # Set title showing symbol, current regime, and bar count
        symbol = self.symbol_var.get().upper()
        regime_names = ['LOW', 'MED', 'HIGH']
        curr_regime = regime_names[bars[-1].regime] if bars else 'N/A'
        self.ax.set_title(f'{symbol} - Regime: {curr_regime} | {len(bars)}/{self.max_bars} bars',
                         color='#c9d1d9', fontsize=12, fontweight='bold')

        self.fig.tight_layout()
        self.canvas.draw_idle()                         # Non-blocking redraw

    def update_stats(self):
        """Update statistics bar with current values and regime confidence."""
        with self.bar_lock:
            bars = list(self.ohlc_bars)
            current = self.current_bar

        if current:
            bars = bars + [current]

        if not bars:
            return

        # Update bar count
        self.stats_labels['Bars'].config(text=str(len(bars)))
        
        # Update session high/low
        all_highs = [b.high for b in bars]
        all_lows = [b.low for b in bars]
        self.stats_labels['High'].config(text=f"{max(all_highs):.2f}")
        self.stats_labels['Low'].config(text=f"{min(all_lows):.2f}")
        
        # Update regime display with color
        regime_names = ['LOW', 'MED', 'HIGH']
        regime_colors = ['#3fb950', '#d29922', '#f85149']
        curr_regime = bars[-1].regime if bars else 0
        self.stats_labels['Regime'].config(
            text=regime_names[curr_regime], 
            fg=regime_colors[curr_regime]
        )
        
        # Show regime confidence (probability)
        prob = self.regime_model.state_probs[curr_regime]
        self.stats_labels['Prob'].config(
            text=f"{prob*100:.1f}%",
            fg=regime_colors[curr_regime]
        )
        
        # Update ticks per bar
        if current:
            self.stats_labels['Ticks/Bar'].config(text=str(current.tick_count))

    def update_trading_ui(self, signal, current_price):
        """Update trading panel with latest signal and position information."""
        # Update position display
        position = self.trading_strategy.position
        if position == 1:
            self.position_label.config(text="LONG", fg='#3fb950')
        elif position == -1:
            self.position_label.config(text="SHORT", fg='#f85149')
        else:
            self.position_label.config(text="FLAT", fg='#8b949e')
        
        # Update P&L
        self.trading_strategy.update_unrealized_pnl(current_price)
        total_pnl = self.trading_strategy.realized_pnl + self.trading_strategy.unrealized_pnl
        pnl_color = '#3fb950' if total_pnl >= 0 else '#f85149'
        self.pnl_label.config(text=f"${total_pnl:.2f}", fg=pnl_color)
        
        # Update signal display
        if signal:
            action = signal.get('action', '--')
            if action == 'BUY':
                self.signal_label.config(text="BUY ↑", fg='#3fb950')
            elif action == 'SELL':
                self.signal_label.config(text="SELL ↓", fg='#f85149')
            
            # Update p-value display
            p_value = signal.get('p_value', None)
            if p_value is not None:
                # Color based on significance
                if p_value < self.trading_strategy.significance_level:
                    pval_color = '#3fb950'  # Significant - green
                else:
                    pval_color = '#d29922'  # Not significant - yellow
                self.pvalue_label.config(text=f"{p_value:.4f}", fg=pval_color)

    def on_closing(self):
        # Clean shutdown when window is closed
        self.running = False
        # Cancel scheduled chart updates to prevent errors
        if hasattr(self, '_after_id'):
            self.root.after_cancel(self._after_id)
        # Disconnect from TWS if connected
        if self.connected:
            try:
                if self.streaming:
                    self.ib_app.cancelMktData(1)
                self.ib_app.disconnect()
            except:
                pass
        self.root.destroy()


# =============================================================================
# APPLICATION ENTRY POINT
# =============================================================================
def main():
    root = tk.Tk()
    app = LiveMarketDashboard(root)
    # Register clean shutdown handler
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()


if __name__ == "__main__":
    main()