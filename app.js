/**
 * Aegis Stock AI - Application Logic
 * Implements:
 * 1. Realistic Stock Price Simulator (GBM + Events)
 * 2. Technical Indicators (SMA, RSI, Volatility)
 * 3. Machine Learning Engines (OLS Linear Regression & Feedforward Neural Network from scratch)
 * 4. Interactive Live Training Loops and Charts (Chart.js)
 * 5. Event-Driven Market Simulators
 */

// GLOBAL APP STATE
const STATE = {
    historicalDays: 200,
    forecastHorizon: 15,
    selectedTicker: 'AAPL',
    selectedModel: 'linear-reg',
    epochs: 150,
    learningRate: 0.05,
    
    // Data stores
    dates: [],
    prices: [],
    highs: [],
    lows: [],
    opens: [],
    volumes: [],
    
    // Technical indicators
    sma5: [],
    sma15: [],
    sma30: [],
    rsi14: [],
    volatility5: [],
    
    // Model evaluations
    testFit: [],
    forecast: [],
    testAccuracyR2: 0,
    testErrorMAE: 0,
    
    // Interactive variables
    newsEventEffect: 0,
    newsEventTimer: 0,
    
    // Chart instances
    priceChart: null,
    lossChart: null,
};

// SYSTEM CONSTANTS FOR TICKER BASELINES
const TICKER_CONFIGS = {
    'AAPL': { base: 175, drift: 0.0003, volatility: 0.012, name: 'Apple Inc.' },
    'TSLA': { base: 220, drift: 0.0006, volatility: 0.024, name: 'Tesla Inc.' },
    'NVDA': { base: 450, drift: 0.0015, volatility: 0.028, name: 'NVIDIA Corp.' },
    'MSFT': { base: 380, drift: 0.0004, volatility: 0.010, name: 'Microsoft Corp.' },
    'BTCUSD': { base: 64000, drift: 0.0008, volatility: 0.035, name: 'Bitcoin / USD' },
    'CUSTOM': { base: 100, drift: 0.0002, volatility: 0.015, name: 'Custom Asset' }
};

// DOM ELEMENTS
const DOM = {
    tickerSelect: document.getElementById('ticker-select'),
    customTickerInput: document.getElementById('custom-ticker-input'),
    historySlider: document.getElementById('history-slider'),
    historyVal: document.getElementById('history-val'),
    horizonSlider: document.getElementById('horizon-slider'),
    horizonVal: document.getElementById('horizon-val'),
    modelType: document.getElementById('model-type'),
    nnParamsWrapper: document.getElementById('nn-params-wrapper'),
    epochsSlider: document.getElementById('epochs-slider'),
    epochsVal: document.getElementById('epochs-val'),
    lrSlider: document.getElementById('lr-slider'),
    lrVal: document.getElementById('lr-val'),
    btnTrain: document.getElementById('btn-train'),
    btnTrainText: document.getElementById('btn-train-text'),
    trainSpinner: document.getElementById('train-spinner'),
    btnRegenerate: document.getElementById('btn-regenerate'),
    
    // Metrics
    metricPrice: document.getElementById('metric-price'),
    metricChange: document.getElementById('metric-change'),
    metricSignal: document.getElementById('metric-signal'),
    metricSignalConfidence: document.getElementById('metric-signal-confidence'),
    metricRsi: document.getElementById('metric-rsi'),
    metricRsiDesc: document.getElementById('metric-rsi-desc'),
    metricAccuracy: document.getElementById('metric-accuracy'),
    metricError: document.getElementById('metric-error'),
    
    // Chart title
    chartMainTitle: document.getElementById('chart-main-title'),
    
    // Terminal & Tabs
    terminalTabs: document.querySelectorAll('.terminal-tab'),
    consoleOutput: document.getElementById('console-output'),
    newsOutput: document.getElementById('news-output'),
    
    // Event Buttons
    eventBtns: document.querySelectorAll('.event-btn'),
    marqueeContent: document.getElementById('marquee-content')
};

// -------------------------------------------------------------
// 1. DATA GENERATOR & SIMULATOR
// -------------------------------------------------------------

function generateStockData() {
    const config = TICKER_CONFIGS[STATE.selectedTicker];
    const days = STATE.historicalDays + 50; // extra padding to compute indicators safely
    
    let S0 = config.base;
    let mu = config.drift;
    let sigma = config.volatility;
    
    // If a custom ticker has a custom input, modify configuration
    if (STATE.selectedTicker === 'CUSTOM' && DOM.customTickerInput.value) {
        const symbol = DOM.customTickerInput.value.toUpperCase();
        // Seed randomness based on ticker letters
        let hash = 0;
        for (let i = 0; i < symbol.length; i++) hash = symbol.charCodeAt(i) + ((hash << 5) - hash);
        S0 = Math.abs(hash % 300) + 50;
        mu = 0.0001 + (hash % 100) / 100000;
        sigma = 0.01 + Math.abs(hash % 50) / 1000;
    }

    const prices = [S0];
    const dates = [];
    const opens = [];
    const highs = [];
    const lows = [];
    const volumes = [];
    
    let currentDate = new Date();
    currentDate.setDate(currentDate.getDate() - Math.floor(days * 1.45)); // Go back enough calendar days to match trading days
    
    let i = 0;
    while (prices.length < days) {
        const dayOfWeek = currentDate.getDay();
        if (dayOfWeek !== 0 && dayOfWeek !== 6) { // Monday-Friday
            const epsilon = randomNormal();
            
            // Apply cycle waves
            const cycle = 0.002 * Math.sin(i / 15.0);
            
            // Apply active news event adjustments
            let eventImpact = 0;
            if (STATE.newsEventTimer > 0) {
                eventImpact = STATE.newsEventEffect / 10.0; // Decay effect distributed
                STATE.newsEventTimer--;
                if (STATE.newsEventTimer === 0) STATE.newsEventEffect = 0;
            }
            
            const dailyReturn = (mu + cycle + eventImpact) + sigma * epsilon;
            const nextPrice = Math.max(1.0, prices[prices.length - 1] * (1 + dailyReturn));
            prices.push(nextPrice);
            
            // High/Low/Open/Vol calculations
            const prevPrice = prices[prices.length - 2];
            const open = prevPrice * (1 + randomNormal() * 0.0025);
            const high = Math.max(open, nextPrice) * (1 + Math.abs(randomNormal() * 0.004));
            const low = Math.min(open, nextPrice) * (1 - Math.abs(randomNormal() * 0.004));
            const vol = Math.floor(Math.exp(12 + randomNormal() * 0.4) * (1 + Math.abs(dailyReturn) * 5));
            
            dates.push(new Date(currentDate));
            opens.push(open);
            highs.push(high);
            lows.push(low);
            volumes.push(vol);
            
            i++;
        }
        currentDate.setDate(currentDate.getDate() + 1);
    }
    
    prices.shift(); // Remove starting S0 seed
    
    STATE.dates = dates;
    STATE.prices = prices;
    STATE.opens = opens;
    STATE.highs = highs;
    STATE.lows = lows;
    STATE.volumes = volumes;
    
    calculateTechnicalIndicators();
}

function calculateTechnicalIndicators() {
    const N = STATE.prices.length;
    
    // Reset technical indicator arrays
    STATE.sma5 = new Array(N).fill(null);
    STATE.sma15 = new Array(N).fill(null);
    STATE.sma30 = new Array(N).fill(null);
    STATE.rsi14 = new Array(N).fill(50.0); // Neutral default
    STATE.volatility5 = new Array(N).fill(0.015); // standard default
    
    // SMA 5, 15, 30
    for (let i = 4; i < N; i++) {
        STATE.sma5[i] = STATE.prices.slice(i - 4, i + 1).reduce((a, b) => a + b, 0) / 5;
    }
    for (let i = 14; i < N; i++) {
        STATE.sma15[i] = STATE.prices.slice(i - 14, i + 1).reduce((a, b) => a + b, 0) / 15;
    }
    for (let i = 29; i < N; i++) {
        STATE.sma30[i] = STATE.prices.slice(i - 29, i + 1).reduce((a, b) => a + b, 0) / 30;
    }
    
    // Volatility 5-day rolling standard deviation of daily return pct
    for (let i = 5; i < N; i++) {
        const returns = [];
        for (let j = i - 4; j <= i; j++) {
            returns.push((STATE.prices[j] - STATE.prices[j - 1]) / STATE.prices[j - 1]);
        }
        const mean = returns.reduce((a, b) => a + b, 0) / 5;
        const sqDiffSum = returns.reduce((a, b) => a + Math.pow(b - mean, 2), 0);
        STATE.volatility5[i] = Math.sqrt(sqDiffSum / 5) || 1e-6;
    }
    
    // RSI 14
    let avgGain = 0;
    let avgLoss = 0;
    
    // Initial RSI computation
    for (let i = 1; i <= 14; i++) {
        const diff = STATE.prices[i] - STATE.prices[i - 1];
        if (diff > 0) avgGain += diff;
        else avgLoss -= diff;
    }
    avgGain /= 14;
    avgLoss /= 14;
    
    if (avgLoss === 0) STATE.rsi14[14] = 100;
    else STATE.rsi14[14] = 100 - (100 / (1 + (avgGain / avgLoss)));
    
    // Smoothing RSI for rest of indices
    for (let i = 15; i < N; i++) {
        const diff = STATE.prices[i] - STATE.prices[i - 1];
        let gain = diff > 0 ? diff : 0;
        let loss = diff < 0 ? -diff : 0;
        
        avgGain = (avgGain * 13 + gain) / 14;
        avgLoss = (avgLoss * 13 + loss) / 14;
        
        if (avgLoss === 0) STATE.rsi14[i] = 100;
        else STATE.rsi14[i] = 100 - (100 / (1 + (avgGain / avgLoss)));
    }
}

// -------------------------------------------------------------
// 2. MATH UTILITIES & SCALING
// -------------------------------------------------------------

function randomNormal() {
    let u = 0, v = 0;
    while(u === 0) u = Math.random(); 
    while(v === 0) v = Math.random();
    return Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v);
}

class StandardScaler {
    constructor() {
        this.means = [];
        this.stds = [];
    }
    
    fit(matrix) {
        const N = matrix.length;
        const P = matrix[0].length;
        this.means = new Array(P).fill(0);
        this.stds = new Array(P).fill(0);
        
        for (let j = 0; j < P; j++) {
            let sum = 0;
            for (let i = 0; i < N; i++) sum += matrix[i][j];
            this.means[j] = sum / N;
            
            let sumSq = 0;
            for (let i = 0; i < N; i++) sumSq += Math.pow(matrix[i][j] - this.means[j], 2);
            this.stds[j] = Math.sqrt(sumSq / N) || 1e-9;
        }
    }
    
    transform(matrix) {
        return matrix.map(row => 
            row.map((val, j) => (val - this.means[j]) / this.stds[j])
        );
    }
    
    transformRow(row) {
        return row.map((val, j) => (val - this.means[j]) / this.stds[j]);
    }
}

// Matrix operations for OLS Normal Equation (X^T X)^-1 X^T y
const MatrixMath = {
    transpose(A) {
        const R = A.length, C = A[0].length;
        const T = Array.from({length: C}, () => new Array(R));
        for (let r = 0; r < R; r++) {
            for (let c = 0; c < C; c++) T[c][r] = A[r][c];
        }
        return T;
    },
    
    multiply(A, B) {
        const rA = A.length, cA = A[0].length;
        const rB = B.length, cB = B[0].length;
        if (cA !== rB) throw new Error("Matrix dimensions mismatch.");
        
        // Handle vector multiplier case
        const out = Array.from({length: rA}, () => new Array(cB).fill(0));
        for (let r = 0; r < rA; r++) {
            for (let c = 0; c < cB; c++) {
                let sum = 0;
                for (let k = 0; k < cA; k++) sum += A[r][k] * B[k][c];
                out[r][c] = sum;
            }
        }
        return out;
    },
    
    // Inverts a small square matrix using Gauss-Jordan Elimination with Pivoting
    invert(A) {
        const n = A.length;
        // Create augmented matrix [A | I]
        const aug = A.map((row, i) => {
            const augmentedRow = [...row];
            for (let j = 0; j < n; j++) augmentedRow.push(i === j ? 1 : 0);
            return augmentedRow;
        });
        
        for (let i = 0; i < n; i++) {
            // Find pivot
            let maxRow = i;
            for (let k = i + 1; k < n; k++) {
                if (Math.abs(aug[k][i]) > Math.abs(aug[maxRow][i])) maxRow = k;
            }
            
            // Swap rows
            const temp = aug[i];
            aug[i] = aug[maxRow];
            aug[maxRow] = temp;
            
            const pivot = aug[i][i];
            if (Math.abs(pivot) < 1e-12) {
                // Non-invertible matrix, return identity default as fallback stabilizer
                return Array.from({length: n}, (_, r) => Array.from({length: n}, (_, c) => r === c ? 1 : 0));
            }
            
            // Normalize pivot row
            for (let j = i; j < 2 * n; j++) aug[i][j] /= pivot;
            
            // Eliminate elements in column i for all other rows
            for (let k = 0; k < n; k++) {
                if (k === i) continue;
                const factor = aug[k][i];
                for (let j = i; j < 2 * n; j++) {
                    aug[k][j] -= factor * aug[i][j];
                }
            }
        }
        
        // Extract right half [I | A^-1]
        return aug.map(row => row.slice(n));
    }
};

// -------------------------------------------------------------
// 3. FEATURE SEPARATION
// -------------------------------------------------------------

function buildMLDataset() {
    // We align data starting from index 30 to make sure indicators (SMA30) are non-null
    const offset = 30;
    const totalSamples = STATE.prices.length - offset - 1; // leave 1 day for next-day target
    
    const features = [];
    const targets = [];
    const indexMapping = []; // map dataset sample i back to STATE index
    
    for (let i = offset; i < STATE.prices.length - 1; i++) {
        features.push([
            STATE.prices[i],               // current close
            STATE.prices[i - 1],           // lag 1
            STATE.prices[i - 2],           // lag 2
            STATE.sma5[i],
            STATE.sma15[i],
            STATE.sma30[i],
            STATE.volatility5[i],
            STATE.rsi14[i] / 100.0         // scale RSI
        ]);
        targets.push(STATE.prices[i + 1]); // Next day's price
        indexMapping.push(i);
    }
    
    return { features, targets, indexMapping };
}

// -------------------------------------------------------------
// 4. MODEL RUNNERS & TRAINERS
// -------------------------------------------------------------

/**
 * Runs Ordinary Least Squares Linear Regression
 */
function runLinearRegression(dataset) {
    const X = dataset.features;
    const y = dataset.targets;
    const N = X.length;
    
    // Train-Test Split (Last 20% for testing)
    const splitIndex = Math.floor(N * 0.8);
    
    const X_train = X.slice(0, splitIndex);
    const y_train = y.slice(0, splitIndex);
    const X_test = X.slice(splitIndex);
    const y_test = y.slice(splitIndex);
    
    writeConsole(`[MODEL] Linear Regression split: ${X_train.length} train, ${X_test.length} test.`, 'info');
    
    // Scale features
    const scaler = new StandardScaler();
    scaler.fit(X_train);
    const X_train_scaled = scaler.transform(X_train);
    const X_test_scaled = scaler.transform(X_test);
    
    // Add bias column (all 1s) to train features
    const X_b = X_train_scaled.map(row => [1, ...row]);
    
    // OLS Normal Equation: Beta = (X^T * X)^-1 * X^T * y
    // Convert y_train to N x 1 matrix
    const Y_matrix = y_train.map(val => [val]);
    
    const Xt = MatrixMath.transpose(X_b);
    const XtX = MatrixMath.multiply(Xt, X_b);
    
    // Add L2 ridge regularization to prevent singular matrix
    for (let r = 0; r < XtX.length; r++) XtX[r][r] += 1e-4;
    
    const XtX_inv = MatrixMath.invert(XtX);
    const XtY = MatrixMath.multiply(Xt, Y_matrix);
    const Beta = MatrixMath.multiply(XtX_inv, XtY); // Beta coefficients (bias + 8 features)
    
    // Calculate Training Loss
    let trainLoss = 0;
    for (let i = 0; i < X_train_scaled.length; i++) {
        let pred = Beta[0][0];
        for (let j = 0; j < X_train_scaled[i].length; j++) {
            pred += X_train_scaled[i][j] * Beta[j + 1][0];
        }
        trainLoss += Math.pow(pred - y_train[i], 2);
    }
    trainLoss /= X_train_scaled.length;
    writeConsole(`[MODEL] OLS optimization complete. Mean Sq Error: ${trainLoss.toFixed(4)}`, 'success');
    
    // Predict Test Data
    const testFits = X_test_scaled.map(row => {
        let pred = Beta[0][0];
        for (let j = 0; j < row.length; j++) pred += row[j] * Beta[j + 1][0];
        return pred;
    });
    
    // Evaluate Test Metrics
    let sumMAE = 0;
    let sumSqDiff = 0;
    let sumTotalVar = 0;
    
    // Compute mean of test actuals
    const meanYTest = y_test.reduce((a, b) => a + b, 0) / y_test.length;
    
    for (let i = 0; i < y_test.length; i++) {
        const error = Math.abs(testFits[i] - y_test[i]);
        sumMAE += error;
        sumSqDiff += Math.pow(testFits[i] - y_test[i], 2);
        sumTotalVar += Math.pow(y_test[i] - meanYTest, 2);
    }
    
    const MAE = sumMAE / y_test.length;
    const R2 = 1 - (sumSqDiff / (sumTotalVar || 1e-9));
    
    STATE.testErrorMAE = MAE;
    STATE.testAccuracyR2 = Math.max(0, R2);
    
    // Populate testFit in global state (aligned to index)
    const testFitAligned = new Array(STATE.prices.length).fill(null);
    const offset = 30;
    for (let i = 0; i < y_test.length; i++) {
        const dataIdx = dataset.indexMapping[splitIndex + i];
        testFitAligned[dataIdx + 1] = testFits[i]; // predicted close for tomorrow (t+1)
    }
    STATE.testFit = testFitAligned;
    
    // --- FORECAST FUTURE ---
    // Start with the last known row and predict step by step
    const forecast = [];
    let currentHistory = STATE.prices.slice();
    let currentSma5 = STATE.sma5.slice();
    let currentSma15 = STATE.sma15.slice();
    let currentSma30 = STATE.sma30.slice();
    let currentVol5 = STATE.volatility5.slice();
    let currentRsi14 = STATE.rsi14.slice();
    
    for (let f = 0; f < STATE.forecastHorizon; f++) {
        const idx = currentHistory.length - 1;
        
        // Prepare feature row for prediction
        const lastFeatures = [
            currentHistory[idx],
            currentHistory[idx - 1],
            currentHistory[idx - 2],
            currentSma5[idx],
            currentSma15[idx],
            currentSma30[idx],
            currentVol5[idx],
            currentRsi14[idx] / 100.0
        ];
        
        // Scale feature row
        const scaledRow = scaler.transformRow(lastFeatures);
        
        // Predict close for tomorrow
        let nextClose = Beta[0][0];
        for (let j = 0; j < scaledRow.length; j++) {
            nextClose += scaledRow[j] * Beta[j + 1][0];
        }
        // bound minimum price to $1.00
        nextClose = Math.max(1.0, nextClose);
        forecast.push(nextClose);
        
        // Update rolling arrays to support autoregressive forecasting
        currentHistory.push(nextClose);
        
        // Recalculate indicators for forecasting step
        const newLength = currentHistory.length;
        currentSma5.push(currentHistory.slice(newLength - 5).reduce((a,b)=>a+b,0) / 5);
        currentSma15.push(currentHistory.slice(newLength - 15).reduce((a,b)=>a+b,0) / 15);
        currentSma30.push(currentHistory.slice(newLength - 30).reduce((a,b)=>a+b,0) / 30);
        
        // Volatility estimation
        const rets = [];
        for (let j = newLength - 5; j < newLength; j++) {
            rets.push((currentHistory[j] - currentHistory[j-1]) / currentHistory[j-1]);
        }
        const m = rets.reduce((a,b)=>a+b,0) / 5;
        const v = Math.sqrt(rets.reduce((a,b)=>a+Math.pow(b-m,2),0) / 5) || 1e-6;
        currentVol5.push(v);
        
        // Keep RSI stable/neutral for projections
        currentRsi14.push(50.0);
    }
    
    STATE.forecast = forecast;
    
    // Draw Static Loss Chart (showing single OLS bar, as OLS is direct calculation)
    updateLossChart([trainLoss], [trainLoss]);
    updateUIElements();
    updateCharts();
}

/**
 * Runs a Feedforward Neural Network (ANN) with Backpropagation
 */
function runNeuralNetwork(dataset) {
    const X = dataset.features;
    const y = dataset.targets;
    const N = X.length;
    
    const splitIndex = Math.floor(N * 0.8);
    const X_train = X.slice(0, splitIndex);
    const y_train = y.slice(0, splitIndex);
    const X_test = X.slice(splitIndex);
    const y_test = y.slice(splitIndex);
    
    // Standard scale features
    const scaler = new StandardScaler();
    scaler.fit(X_train);
    const X_train_scaled = scaler.transform(X_train);
    const X_test_scaled = scaler.transform(X_test);
    
    // Network Dimensions: Input P=8, Hidden H=10, Output O=1
    const P = 8;
    const H = 10;
    const O = 1;
    
    // Initialize Weights and Biases (He / Xavier Initialization)
    let W1 = Array.from({length: P}, () => Array.from({length: H}, () => randomNormal() * Math.sqrt(2 / P)));
    let B1 = new Array(H).fill(0);
    let W2 = Array.from({length: H}, () => Array.from({length: O}, () => randomNormal() * Math.sqrt(2 / H)));
    let B2 = new Array(O).fill(0);
    
    // Training configurations
    const epochs = STATE.epochs;
    const lr = STATE.learningRate;
    
    const lossHistory = [];
    const valLossHistory = [];
    
    let epoch = 0;
    
    // Disable UI interactions during NN training
    DOM.btnTrain.disabled = true;
    DOM.btnTrainText.textContent = "Training Network...";
    DOM.trainSpinner.classList.remove('hidden');
    
    // Run training chunked in requestAnimationFrame to animate convergence live without locking browser thread!
    function trainStep() {
        if (epoch >= epochs) {
            // End of training
            finishNNTraining(W1, B1, W2, B2, X_test_scaled, y_test, scaler, dataset, splitIndex);
            return;
        }
        
        // Run 5 training steps per frame to speed up visuals
        for (let step = 0; step < 5 && epoch < epochs; step++) {
            // Feedforward (Batch size = full training dataset)
            const Z1 = []; // N x H
            const A1 = []; // N x H (ReLU Activation)
            const Z2 = []; // N x 1
            const y_pred = []; // N
            
            for (let i = 0; i < X_train_scaled.length; i++) {
                const rowX = X_train_scaled[i];
                
                // Input -> Hidden
                const rowZ1 = [];
                const rowA1 = [];
                for (let h = 0; h < H; h++) {
                    let sum = B1[h];
                    for (let p = 0; p < P; p++) sum += rowX[p] * W1[p][h];
                    rowZ1.push(sum);
                    rowA1.push(Math.max(0, sum)); // ReLU
                }
                Z1.push(rowZ1);
                A1.push(rowA1);
                
                // Hidden -> Output
                let sumZ2 = B2[0];
                for (let h = 0; h < H; h++) sumZ2 += rowA1[h] * W2[h][0];
                Z2.push([sumZ2]);
                y_pred.push(sumZ2);
            }
            
            // Calculate Mean Squared Error Loss
            let mse = 0;
            for (let i = 0; i < y_pred.length; i++) mse += Math.pow(y_pred[i] - y_train[i], 2);
            mse /= y_pred.length;
            lossHistory.push(mse);
            
            // Compute Validation loss on Test set
            let testMse = 0;
            for (let i = 0; i < X_test_scaled.length; i++) {
                const rowX = X_test_scaled[i];
                const rowA1 = [];
                for (let h = 0; h < H; h++) {
                    let sum = B1[h];
                    for (let p = 0; p < P; p++) sum += rowX[p] * W1[p][h];
                    rowA1.push(Math.max(0, sum));
                }
                let pred = B2[0];
                for (let h = 0; h < H; h++) pred += rowA1[h] * W2[h][0];
                testMse += Math.pow(pred - y_test[i], 2);
            }
            testMse /= X_test_scaled.length;
            valLossHistory.push(testMse);
            
            // BACKPROPAGATION
            const dZ2 = []; // N x 1
            for (let i = 0; i < y_pred.length; i++) dZ2.push([y_pred[i] - y_train[i]]);
            
            // Gradient Hidden to Output
            const dW2 = Array.from({length: H}, () => new Array(O).fill(0));
            let dB2 = [0];
            for (let i = 0; i < X_train_scaled.length; i++) {
                dB2[0] += dZ2[i][0];
                for (let h = 0; h < H; h++) dW2[h][0] += A1[i][h] * dZ2[i][0];
            }
            dB2[0] /= X_train_scaled.length;
            for (let h = 0; h < H; h++) dW2[h][0] /= X_train_scaled.length;
            
            // Gradient Hidden layer
            const dZ1 = Array.from({length: X_train_scaled.length}, () => new Array(H).fill(0));
            for (let i = 0; i < X_train_scaled.length; i++) {
                for (let h = 0; h < H; h++) {
                    let dHiddenVal = dZ2[i][0] * W2[h][0];
                    // Derivative of ReLU
                    dZ1[i][h] = Z1[i][h] > 0 ? dHiddenVal : 0;
                }
            }
            
            // Gradient Input to Hidden
            const dW1 = Array.from({length: P}, () => new Array(H).fill(0));
            const dB1 = new Array(H).fill(0);
            for (let i = 0; i < X_train_scaled.length; i++) {
                for (let h = 0; h < H; h++) {
                    dB1[h] += dZ1[i][h];
                    for (let p = 0; p < P; p++) dW1[p][h] += X_train_scaled[i][p] * dZ1[i][h];
                }
            }
            
            for (let h = 0; h < H; h++) {
                dB1[h] /= X_train_scaled.length;
                for (let p = 0; p < P; p++) dW1[p][h] /= X_train_scaled.length;
            }
            
            // WEIGHTS UPDATE (SGD with momentum simplified to base gradient descent)
            for (let p = 0; p < P; p++) {
                for (let h = 0; h < H; h++) W1[p][h] -= lr * dW1[p][h];
            }
            for (let h = 0; h < H; h++) B1[h] -= lr * dB1[h];
            for (let h = 0; h < H; h++) W2[h][0] -= lr * dW2[h][0];
            B2[0] -= lr * dB2[0];
            
            epoch++;
        }
        
        // Visual logging progress
        if (epoch % 20 === 0 || epoch === epochs) {
            writeConsole(`[TRAINING] Epoch ${epoch}/${epochs} - Loss: ${lossHistory[lossHistory.length-1].toFixed(4)} | Val Loss: ${valLossHistory[valLossHistory.length-1].toFixed(4)}`, 'epoch');
            updateLossChart(lossHistory, valLossHistory);
        }
        
        requestAnimationFrame(trainStep);
    }
    
    // Start loop
    trainStep();
}

function finishNNTraining(W1, B1, W2, B2, X_test_scaled, y_test, scaler, dataset, splitIndex) {
    DOM.btnTrain.disabled = false;
    DOM.btnTrainText.textContent = "Run AI Predictor";
    DOM.trainSpinner.classList.add('hidden');
    
    // Evaluate Test Data Predictions
    const testFits = [];
    for (let i = 0; i < X_test_scaled.length; i++) {
        const rowX = X_test_scaled[i];
        const rowA1 = [];
        for (let h = 0; h < W2.length; h++) {
            let sum = B1[h];
            for (let p = 0; p < rowX.length; p++) sum += rowX[p] * W1[p][h];
            rowA1.push(Math.max(0, sum));
        }
        let pred = B2[0];
        for (let h = 0; h < W2.length; h++) pred += rowA1[h] * W2[h][0];
        testFits.push(pred);
    }
    
    // Calculate metrics
    let sumMAE = 0;
    let sumSqDiff = 0;
    let sumTotalVar = 0;
    const meanYTest = y_test.reduce((a, b) => a + b, 0) / y_test.length;
    
    for (let i = 0; i < y_test.length; i++) {
        const error = Math.abs(testFits[i] - y_test[i]);
        sumMAE += error;
        sumSqDiff += Math.pow(testFits[i] - y_test[i], 2);
        sumTotalVar += Math.pow(y_test[i] - meanYTest, 2);
    }
    
    const MAE = sumMAE / y_test.length;
    const R2 = 1 - (sumSqDiff / (sumTotalVar || 1e-9));
    
    STATE.testErrorMAE = MAE;
    STATE.testAccuracyR2 = Math.max(0, R2);
    
    writeConsole(`[MODEL] NN training complete. Final R² score: ${(STATE.testAccuracyR2 * 100).toFixed(2)}%`, 'success');
    
    // Map fits to aligned global state
    const testFitAligned = new Array(STATE.prices.length).fill(null);
    for (let i = 0; i < y_test.length; i++) {
        const dataIdx = dataset.indexMapping[splitIndex + i];
        testFitAligned[dataIdx + 1] = testFits[i];
    }
    STATE.testFit = testFitAligned;
    
    // FORECAST FUTURE AUTOREGRESSIVELY
    const forecast = [];
    let currentHistory = STATE.prices.slice();
    let currentSma5 = STATE.sma5.slice();
    let currentSma15 = STATE.sma15.slice();
    let currentSma30 = STATE.sma30.slice();
    let currentVol5 = STATE.volatility5.slice();
    let currentRsi14 = STATE.rsi14.slice();
    
    for (let f = 0; f < STATE.forecastHorizon; f++) {
        const idx = currentHistory.length - 1;
        const lastFeatures = [
            currentHistory[idx],
            currentHistory[idx - 1],
            currentHistory[idx - 2],
            currentSma5[idx],
            currentSma15[idx],
            currentSma30[idx],
            currentVol5[idx],
            currentRsi14[idx] / 100.0
        ];
        
        const scaledRow = scaler.transformRow(lastFeatures);
        
        // Feedforward single step
        const rowA1 = [];
        for (let h = 0; h < W2.length; h++) {
            let sum = B1[h];
            for (let p = 0; p < scaledRow.length; p++) sum += scaledRow[p] * W1[p][h];
            rowA1.push(Math.max(0, sum));
        }
        let nextClose = B2[0];
        for (let h = 0; h < W2.length; h++) nextClose += rowA1[h] * W2[h][0];
        
        nextClose = Math.max(1.0, nextClose);
        forecast.push(nextClose);
        
        // Rolling update
        currentHistory.push(nextClose);
        const newLength = currentHistory.length;
        currentSma5.push(currentHistory.slice(newLength - 5).reduce((a,b)=>a+b,0) / 5);
        currentSma15.push(currentHistory.slice(newLength - 15).reduce((a,b)=>a+b,0) / 15);
        currentSma30.push(currentHistory.slice(newLength - 30).reduce((a,b)=>a+b,0) / 30);
        
        const rets = [];
        for (let j = newLength - 5; j < newLength; j++) {
            rets.push((currentHistory[j] - currentHistory[j-1]) / currentHistory[j-1]);
        }
        const m = rets.reduce((a,b)=>a+b,0) / 5;
        currentVol5.push(Math.sqrt(rets.reduce((a,b)=>a+Math.pow(b-m,2),0) / 5) || 1e-6);
        currentRsi14.push(50.0);
    }
    
    STATE.forecast = forecast;
    updateUIElements();
    updateCharts();
}

/**
 * Runs a simple Technical SMA / Momentum crossover engine
 */
function runMovingAveragePredictor(dataset) {
    const y = dataset.targets;
    const N = y.length;
    const splitIndex = Math.floor(N * 0.8);
    const y_test = y.slice(splitIndex);
    
    writeConsole(`[MODEL] Running SMA / Momentum crossover algorithm...`, 'info');
    
    // SMA predictor behaves by checking the direction of the SMA crossover
    const testFits = [];
    for (let i = splitIndex; i < N; i++) {
        const dataIdx = dataset.indexMapping[i];
        
        // Momentum crossover prediction: Close + (SMA5 - SMA15) * scale
        const sma5Val = STATE.sma5[dataIdx];
        const sma15Val = STATE.sma15[dataIdx];
        const diff = sma5Val - sma15Val;
        
        // Project tomorrow is current close * drift factor from Crossover
        const multiplier = 1 + (diff / (sma15Val || 1)) * 0.5;
        testFits.push(STATE.prices[dataIdx] * multiplier);
    }
    
    // Metrics
    let sumMAE = 0;
    let sumSqDiff = 0;
    let sumTotalVar = 0;
    const meanYTest = y_test.reduce((a, b) => a + b, 0) / y_test.length;
    
    for (let i = 0; i < y_test.length; i++) {
        const error = Math.abs(testFits[i] - y_test[i]);
        sumMAE += error;
        sumSqDiff += Math.pow(testFits[i] - y_test[i], 2);
        sumTotalVar += Math.pow(y_test[i] - meanYTest, 2);
    }
    
    STATE.testErrorMAE = sumMAE / y_test.length;
    STATE.testAccuracyR2 = Math.max(0, 1 - (sumSqDiff / (sumTotalVar || 1e-9)));
    
    const testFitAligned = new Array(STATE.prices.length).fill(null);
    for (let i = 0; i < y_test.length; i++) {
        const dataIdx = dataset.indexMapping[splitIndex + i];
        testFitAligned[dataIdx + 1] = testFits[i];
    }
    STATE.testFit = testFitAligned;
    
    // Forecast future (extrapolating momentum drift)
    const forecast = [];
    const lastPrice = STATE.prices[STATE.prices.length - 1];
    const lastSma5 = STATE.sma5[STATE.prices.length - 1];
    const lastSma15 = STATE.sma15[STATE.prices.length - 1];
    const momentumDrift = (lastSma5 - lastSma15) / (lastSma15 || 1);
    
    for (let f = 0; f < STATE.forecastHorizon; f++) {
        // Project decaying momentum
        const decay = Math.pow(0.9, f);
        const forecastPrice = lastPrice * (1 + momentumDrift * 0.1 * decay);
        forecast.push(forecastPrice);
    }
    
    STATE.forecast = forecast;
    
    // Draw Loss Chart with baseline dummy
    updateLossChart([STATE.testErrorMAE], [STATE.testErrorMAE]);
    writeConsole(`[MODEL] Momentum prediction fit complete. MAE: $${STATE.testErrorMAE.toFixed(2)}`, 'success');
    
    updateUIElements();
    updateCharts();
}

function runPipeline() {
    const dataset = buildMLDataset();
    if (dataset.features.length < 50) {
        writeConsole("[ERROR] Too few historical data points to feed features. Adjust Historical range.", 'error');
        return;
    }
    
    writeConsole(`[SYSTEM] Initialized data pipeline. Size: ${dataset.features.length} samples.`, 'system');
    
    if (STATE.selectedModel === 'linear-reg') {
        runLinearRegression(dataset);
    } else if (STATE.selectedModel === 'neural-net') {
        runNeuralNetwork(dataset);
    } else if (STATE.selectedModel === 'moving-average') {
        runMovingAveragePredictor(dataset);
    }
}

// -------------------------------------------------------------
// 5. CHART RENDERERS (CHART.JS)
// -------------------------------------------------------------

function initCharts() {
    // 1. PRICE & FORECAST CHART
    const ctxPrice = document.getElementById('priceChart').getContext('2d');
    
    // Gradient definitions
    const primaryGlow = ctxPrice.createLinearGradient(0, 0, 0, 400);
    primaryGlow.addColorStop(0, 'rgba(6, 182, 212, 0.25)');
    primaryGlow.addColorStop(1, 'rgba(6, 182, 212, 0.0)');
    
    STATE.priceChart = new Chart(ctxPrice, {
        type: 'line',
        data: {
            labels: [],
            datasets: [
                {
                    label: 'Actual Price',
                    data: [],
                    borderColor: '#06b6d4',
                    borderWidth: 2.5,
                    backgroundColor: primaryGlow,
                    fill: true,
                    pointRadius: 0,
                    tension: 0.1
                },
                {
                    label: 'Model Test Fit',
                    data: [],
                    borderColor: '#f43f5e',
                    borderWidth: 1.5,
                    borderDash: [4, 4],
                    pointRadius: 0,
                    fill: false,
                    tension: 0.1
                },
                {
                    label: 'Future Forecast',
                    data: [],
                    borderColor: '#f59e0b',
                    borderWidth: 3,
                    backgroundColor: 'rgba(245, 158, 11, 0.05)',
                    fill: false,
                    pointRadius: 3,
                    pointBackgroundColor: '#f59e0b',
                    tension: 0.15
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { display: false } // Custom legend is used in UI HTML
            },
            scales: {
                x: {
                    grid: { color: 'rgba(255, 255, 255, 0.04)', borderColor: 'rgba(255,255,255,0.1)' },
                    ticks: { color: '#64748b', font: { family: 'Outfit', size: 10 } }
                },
                y: {
                    grid: { color: 'rgba(255, 255, 255, 0.04)', borderColor: 'rgba(255,255,255,0.1)' },
                    ticks: { color: '#64748b', font: { family: 'Space Grotesk', size: 11 } }
                }
            }
        }
    });

    // 2. CONVERGENCE LOSS CHART
    const ctxLoss = document.getElementById('lossChart').getContext('2d');
    STATE.lossChart = new Chart(ctxLoss, {
        type: 'line',
        data: {
            labels: [],
            datasets: [
                {
                    label: 'Training Loss',
                    data: [],
                    borderColor: '#6366f1',
                    borderWidth: 2,
                    fill: false,
                    pointRadius: 0,
                    tension: 0.1
                },
                {
                    label: 'Validation Loss',
                    data: [],
                    borderColor: '#fb7185',
                    borderWidth: 1.5,
                    fill: false,
                    pointRadius: 0,
                    tension: 0.1
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    labels: { color: '#94a3b8', font: { family: 'Outfit', size: 10 } }
                }
            },
            scales: {
                x: {
                    grid: { display: false },
                    ticks: { color: '#64748b', font: { size: 9 } }
                },
                y: {
                    type: 'logarithmic', // Use log scale since NN loss falls exponentially
                    grid: { color: 'rgba(255, 255, 255, 0.03)' },
                    ticks: { color: '#64748b', font: { size: 9 } }
                }
            }
        }
    });
}

function updateCharts() {
    if (!STATE.priceChart) return;
    
    // Prepare Labels (Dates formatted as Mon DD)
    const formattedDates = STATE.dates.map(d => {
        return d.toLocaleDateString('en-US', { month: 'short', day: 'numeric' });
    });
    
    // Forecast Labels
    const lastDate = STATE.dates[STATE.dates.length - 1];
    const forecastDates = [];
    for (let i = 1; i <= STATE.forecastHorizon; i++) {
        const nextDate = new Date(lastDate);
        nextDate.setDate(nextDate.getDate() + i);
        forecastDates.push(nextDate.toLocaleDateString('en-US', { month: 'short', day: 'numeric' }));
    }
    
    // Combine Labels: Historical Dates + Forecast Dates
    const allLabels = [...formattedDates, ...forecastDates];
    
    // Combine Datasets:
    // Actuals are padded with nulls over the forecast window
    const actualsPadded = [...STATE.prices];
    for (let i = 0; i < STATE.forecastHorizon; i++) actualsPadded.push(null);
    
    // Fits are padded with nulls over forecast window
    const fitsPadded = [...STATE.testFit];
    for (let i = 0; i < STATE.forecastHorizon; i++) fitsPadded.push(null);
    
    // Forecasts is padded with nulls over actual history, and connects to the last actual price
    const forecastPadded = new Array(STATE.prices.length - 1).fill(null);
    forecastPadded.push(STATE.prices[STATE.prices.length - 1]); // Connect point
    forecastPadded.push(...STATE.forecast);
    
    // Update chart data
    STATE.priceChart.data.labels = allLabels;
    STATE.priceChart.data.datasets[0].data = actualsPadded;
    STATE.priceChart.data.datasets[1].data = fitsPadded;
    STATE.priceChart.data.datasets[2].data = forecastPadded;
    
    // Dynamic styling changes
    STATE.priceChart.update();
}

function updateLossChart(trainLoss, valLoss) {
    if (!STATE.lossChart) return;
    
    STATE.lossChart.data.labels = trainLoss.map((_, i) => i);
    STATE.lossChart.data.datasets[0].data = trainLoss;
    STATE.lossChart.data.datasets[1].data = valLoss;
    
    // Switch to logarithmic scale if training network, normal scale otherwise
    if (STATE.selectedModel === 'neural-net') {
        STATE.lossChart.options.scales.y.type = 'logarithmic';
    } else {
        STATE.lossChart.options.scales.y.type = 'linear';
    }
    
    STATE.lossChart.update();
}

// -------------------------------------------------------------
// 6. UI RENDER CONTROLLER
// -------------------------------------------------------------

function updateUIElements() {
    // Current price & Day-over-day Change
    const currentPrice = STATE.prices[STATE.prices.length - 1];
    const prevPrice = STATE.prices[STATE.prices.length - 2];
    const pctChange = ((currentPrice - prevPrice) / prevPrice) * 100;
    
    DOM.metricPrice.textContent = `$${currentPrice.toFixed(2)}`;
    DOM.metricChange.textContent = `${pctChange >= 0 ? '+' : ''}${pctChange.toFixed(2)}%`;
    DOM.metricChange.className = `metric-change ${pctChange >= 0 ? 'positive' : 'negative'}`;
    
    // RSI Updates
    const currentRsi = STATE.rsi14[STATE.rsi14.length - 1];
    DOM.metricRsi.textContent = currentRsi.toFixed(1);
    
    if (currentRsi >= 70) {
        DOM.metricRsiDesc.textContent = "Overbought (Bearish Crossover)";
        DOM.metricRsiDesc.className = "metric-desc text-danger";
    } else if (currentRsi <= 30) {
        DOM.metricRsiDesc.textContent = "Oversold (Bullish Signal)";
        DOM.metricRsiDesc.className = "metric-desc text-success";
    } else {
        DOM.metricRsiDesc.textContent = "Neutral Momentum";
        DOM.metricRsiDesc.className = "metric-desc text-neutral";
    }
    
    // Accuracy
    DOM.metricAccuracy.textContent = `${(STATE.testAccuracyR2 * 100).toFixed(1)}%`;
    DOM.metricError.textContent = `MAE: $${STATE.testErrorMAE.toFixed(2)}`;
    
    // AI Forecast Gauge Signal direction
    const forecastStart = currentPrice;
    const forecastEnd = STATE.forecast[STATE.forecast.length - 1] || currentPrice;
    const forecastDiff = ((forecastEnd - forecastStart) / forecastStart) * 100;
    
    // Confidence calculation based on model R2 score and movement magnitude
    const baseConf = 60 + Math.min(30, Math.floor(STATE.testAccuracyR2 * 30));
    const finalConf = Math.min(98, Math.max(50, Math.round(baseConf + Math.abs(forecastDiff) * 2)));
    DOM.metricSignalConfidence.textContent = `${finalConf}% Conf`;
    
    if (forecastDiff > 1.5) {
        DOM.metricSignal.textContent = "BUY";
        DOM.metricSignal.className = "metric-val signal-val BUY";
    } else if (forecastDiff < -1.5) {
        DOM.metricSignal.textContent = "SELL";
        DOM.metricSignal.className = "metric-val signal-val SELL";
    } else {
        DOM.metricSignal.textContent = "HOLD";
        DOM.metricSignal.className = "metric-val signal-val HOLD";
    }
    
    // Update chart title symbol
    let name = STATE.selectedTicker;
    if (STATE.selectedTicker === 'CUSTOM') {
        name = DOM.customTickerInput.value.toUpperCase() || 'CUSTOM';
    }
    DOM.chartMainTitle.textContent = `${name} Stock Price vs AI Forecast`;
}

function writeConsole(text, type = 'default') {
    const line = document.createElement('div');
    line.className = `terminal-line ${type}`;
    line.innerHTML = text;
    DOM.consoleOutput.appendChild(line);
    
    // Autoscroll
    DOM.consoleOutput.scrollTop = DOM.consoleOutput.scrollHeight;
}

function writeNews(headline, time = null) {
    if (!time) {
        const now = new Date();
        time = now.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
    }
    
    const line = document.createElement('div');
    line.className = 'terminal-line news-item';
    line.innerHTML = `<span class="news-time">${time}</span> - ${headline}`;
    DOM.newsOutput.insertBefore(line, DOM.newsOutput.firstChild);
}

// -------------------------------------------------------------
// 7. SYSTEM INTERACTIONS & HANDLERS
// -------------------------------------------------------------

function setupEventListeners() {
    // Ticker selector dropdown
    DOM.tickerSelect.addEventListener('change', (e) => {
        STATE.selectedTicker = e.target.value;
        if (STATE.selectedTicker === 'CUSTOM') {
            DOM.customTickerInput.classList.add('active');
        } else {
            DOM.customTickerInput.classList.remove('active');
        }
        
        writeConsole(`[SYSTEM] Switched ticker focus to ${STATE.selectedTicker}. Generating data...`, 'system');
        generateStockData();
        runPipeline();
    });
    
    DOM.customTickerInput.addEventListener('change', () => {
        generateStockData();
        runPipeline();
    });
    
    // Sliders
    DOM.historySlider.addEventListener('input', (e) => {
        STATE.historicalDays = parseInt(e.target.value);
        DOM.historyVal.textContent = `${STATE.historicalDays} days`;
    });
    DOM.historySlider.addEventListener('change', () => {
        generateStockData();
        runPipeline();
    });
    
    DOM.horizonSlider.addEventListener('input', (e) => {
        STATE.forecastHorizon = parseInt(e.target.value);
        DOM.horizonVal.textContent = `${STATE.forecastHorizon} days`;
    });
    DOM.horizonSlider.addEventListener('change', () => {
        runPipeline();
    });
    
    // Model Type Dropdown
    DOM.modelType.addEventListener('change', (e) => {
        STATE.selectedModel = e.target.value;
        if (STATE.selectedModel === 'neural-net') {
            DOM.nnParamsWrapper.style.display = 'block';
        } else {
            DOM.nnParamsWrapper.style.display = 'none';
        }
        writeConsole(`[SYSTEM] AI architecture updated to: ${STATE.selectedModel.toUpperCase()}`, 'system');
    });
    
    DOM.epochsSlider.addEventListener('input', (e) => {
        STATE.epochs = parseInt(e.target.value);
        DOM.epochsVal.textContent = STATE.epochs;
    });
    
    DOM.lrSlider.addEventListener('input', (e) => {
        STATE.learningRate = parseFloat(e.target.value);
        DOM.lrVal.textContent = STATE.learningRate.toFixed(3);
    });
    
    // Buttons
    DOM.btnTrain.addEventListener('click', () => {
        runPipeline();
    });
    
    DOM.btnRegenerate.addEventListener('click', () => {
        writeConsole('[SYSTEM] Purging current stock buffer. Recalibrating Geometric Brownian volatility parameters.', 'system');
        generateStockData();
        runPipeline();
    });
    
    // Terminal Tab switching
    DOM.terminalTabs.forEach(tab => {
        tab.addEventListener('click', () => {
            DOM.terminalTabs.forEach(t => t.classList.remove('active'));
            tab.classList.add('active');
            
            const target = tab.getAttribute('data-target');
            document.querySelectorAll('.terminal-pane').forEach(pane => {
                pane.classList.remove('active');
            });
            document.getElementById(target).classList.add('active');
        });
    });
    
    // Event injector buttons
    DOM.eventBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            const eventType = btn.getAttribute('data-event');
            triggerMarketEvent(eventType);
        });
    });
}

function triggerMarketEvent(type) {
    let headline = "";
    let impact = 0;
    
    switch (type) {
        case 'earnings':
            headline = `${STATE.selectedTicker} reports record Q3 profits: EPS beats estimates by 15.4% as cloud revenue surges.`;
            impact = 0.08;
            break;
        case 'regulatory':
            headline = `Breaking: Antitrust investigators launch formal probe into ${STATE.selectedTicker} market practices.`;
            impact = -0.12;
            break;
        case 'hype':
            headline = `CEO posts cryptic late-night tweet hinting at massive product breakthroughs next week. Speculators rush buy order blocks.`;
            impact = 0.05;
            break;
        case 'inflation':
            headline = `Bureau of Labor Statistics updates CPI core inflation indexes higher than forecast. Markets plunge broadly.`;
            impact = -0.07;
            break;
    }
    
    writeNews(headline);
    writeConsole(`[EVENT INJECTED] Triggered event: ${type.toUpperCase()}. Recalibrating drift coeff by ${(impact * 100).toFixed(1)}%.`, 'warning');
    
    // Apply structural drift change for the next few simulated steps
    STATE.newsEventEffect = impact;
    STATE.newsEventTimer = 15; // steps
    
    // Generate new stock data appending the event, and retrain pipeline
    generateStockData();
    runPipeline();
    
    // Visual flash animation on metrics
    const priceCard = DOM.metricPrice.closest('.metric-card');
    priceCard.style.transition = 'none';
    priceCard.style.boxShadow = impact > 0 ? '0 0 25px rgba(16, 185, 129, 0.4)' : '0 0 25px rgba(244, 63, 94, 0.4)';
    setTimeout(() => {
        priceCard.style.transition = 'var(--transition-smooth)';
        priceCard.style.boxShadow = '';
    }, 400);
}

// -------------------------------------------------------------
// 8. TICKER MARQUEE ROTATOR
// -------------------------------------------------------------

function initMarquee() {
    const defaultTickers = [
        { sym: 'SPY', base: 512.4, change: 0.25 },
        { sym: 'DIA', base: 390.1, change: -0.12 },
        { sym: 'QQQ', base: 438.8, change: 0.65 },
        { sym: 'IWM', base: 202.4, change: -0.42 },
        { sym: 'BTC', base: 64850, change: 2.15 },
        { sym: 'ETH', base: 3480, change: 1.85 },
        { sym: 'EURUSD', base: 1.085, change: -0.05 }
    ];
    
    // Render marquee HTML twice for infinite scroll loop
    const render = () => {
        let html = "";
        defaultTickers.forEach(t => {
            const diff = t.base * (t.change / 100);
            t.base += diff * (Math.random() - 0.5) * 0.1; // small random movements
            const colorClass = t.change >= 0 ? 'text-success' : 'text-danger';
            const arrow = t.change >= 0 ? '▲' : '▼';
            html += `
                <div class="marquee-ticker">
                    <span class="symbol">${t.sym}</span>
                    <span class="price">$${t.base.toLocaleString(undefined, {minimumFractionDigits: 2, maximumFractionDigits: 2})}</span>
                    <span class="change ${colorClass}">${arrow} ${Math.abs(t.change).toFixed(2)}%</span>
                </div>
            `;
        });
        return html;
    };
    
    const elements = render();
    DOM.marqueeContent.innerHTML = elements + elements;
    
    // Tick movements every 3 seconds
    setInterval(() => {
        defaultTickers.forEach(t => {
            const movement = (Math.random() - 0.5) * 0.05;
            t.change += movement;
            t.base = t.base * (1 + movement / 100);
        });
        const updated = render();
        DOM.marqueeContent.innerHTML = updated + updated;
    }, 3000);
}

// -------------------------------------------------------------
// 9. APP INITIALIZATION
// -------------------------------------------------------------

window.addEventListener('DOMContentLoaded', () => {
    initCharts();
    initMarquee();
    setupEventListeners();
    
    // Run initial data generate & pipeline execute
    generateStockData();
    runPipeline();
    
    writeConsole('[SYSTEM] Application fully loaded. All models ready for inference.', 'system');
});
