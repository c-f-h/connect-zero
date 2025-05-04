class Connect4Game {
    constructor(gameContainer) {
        // Game constants
        this.ROWS = 6;
        this.COLS = 7;
        this.CELL_SIZE = 60;
        this.EMPTY = 0;
        this.PLAYER1 = 1;
        this.PLAYER2 = 2;
        this.COLORS = {
            [this.PLAYER1]: '#ff4d4d',
            [this.PLAYER2]: '#ffd700',
            [this.EMPTY]: '#f0f0f0'
        };
        this.humanPlayer = -1;
        this.cpuPlayer = -1;
        this.onnxModel = null;

        // Game variables
        this.board = [];
        this.gameOver = false;
        this.moveHistory = [];
        this.currentMoveIndex = -1;

        this.createGameContainerContent(gameContainer);
        this.init(gameContainer);
    }

    createGameContainerContent(gameContainer) {
        gameContainer.innerHTML = `
            <div class="status"></div>
            <div class="game-layout">
                <svg class="board" viewBox="0 0 420 360"></svg>
                <div class="valuation-bar">
                    <div class="valuation-center-line"></div>
                    <div class="valuation-indicator"></div>
                </div>
            </div>
            <div class="controls">
                <button class="firstButton">⏮</button>
                <button class="backButton">◀</button>
                <span class="moveCounter">Move 0 / 0</span>
                <button class="forwardButton">▶</button>
                <button class="lastButton">⏭</button>
                <span></span>
                <button class="resetButton">Reset Game</button>
            </div>
        `;
    }

    init(gameContainer) {
        this.gameContainer = gameContainer;
        this.boardSvg = this.gameContainer.querySelector('.board');
        this.statusDiv = this.gameContainer.querySelector('.status');
        this.backButton = this.gameContainer.querySelector('.backButton');
        this.forwardButton = this.gameContainer.querySelector('.forwardButton');
        this.firstButton = this.gameContainer.querySelector('.firstButton');
        this.lastButton = this.gameContainer.querySelector('.lastButton');
        this.moveCounter = this.gameContainer.querySelector('.moveCounter');
        this.resetButton = this.gameContainer.querySelector('.resetButton');
        this.valuationBar = this.gameContainer.querySelector('.valuation-bar');
        this.valuationIndicator = this.gameContainer.querySelector('.valuation-indicator');

        this.resetButton.addEventListener('click', () => this.resetGame());
        this.backButton.addEventListener('click', () => this.navigateBack());
        this.forwardButton.addEventListener('click', () => this.navigateForward());
        this.firstButton.addEventListener('click', () => this.navigateFirst());
        this.lastButton.addEventListener('click', () => this.navigateLast());

        if (this.gameContainer.dataset.human)
            this.humanPlayer = parseInt(this.gameContainer.dataset.human);
        if (this.gameContainer.dataset.cpu)
            this.cpuPlayer = parseInt(this.gameContainer.dataset.cpu);
        if (this.gameContainer.dataset.randomFirstPlayer)
            this.randomFirstPlayer = this.gameContainer.dataset.randomFirstPlayer === 'true';

        if (this.gameContainer.dataset.onnxModel) {
            this.onnxModel = new Connect4ONNXModel(this.gameContainer.dataset.onnxModel);
            this.onnxModel.initialize().then(() => {
                console.log("Model initialized successfully.");
                this.resetGame();
            }).catch(error => {
                console.error("Error initializing model:", error);
            });
        }
        else {
            this.valuationBar.style.display = 'none';
            this.resetGame(); // No need to wait for model initialization
        }
    }

    resetGame() {
        if (this.randomFirstPlayer && Math.random() < 0.5) {
            [this.humanPlayer, this.cpuPlayer] = [this.cpuPlayer, this.humanPlayer];
        }

        this.board = Array(this.ROWS).fill().map(() => Array(this.COLS).fill(this.EMPTY));

        if (this.gameContainer.dataset.movelist) {
            this.moveHistory = JSON.parse(this.gameContainer.dataset.movelist).map(move => parseInt(move));
            this.resetButton.style.display = 'none';
        } else
            this.moveHistory = [];
        this.currentMoveIndex = 0;
        this.checkForGameOver();
        this.boardSvg.innerHTML = '';
        this.drawBoard();
        this.setValuation(0);
        this.onBoardStateChanged();
        this.checkIfComputerShouldMove();
    }

    drawBoard() {
        const boardBackground = this.createSvgElement('rect', {
            class: 'board-background',
            x: 0,
            y: 0,
            width: this.COLS * this.CELL_SIZE,
            height: this.ROWS * this.CELL_SIZE,
        });
        this.boardSvg.appendChild(boardBackground);

        const columnGroups = [];
        for (let col = 0; col < this.COLS; col++) {
            const colGroup = this.createSvgElement('g');
            colGroup.onclick = () => this.handleColumnClick(col);
            columnGroups.push(colGroup);
            this.boardSvg.appendChild(colGroup);

            const hoverArea = this.createSvgElement('rect', {
                x: col * this.CELL_SIZE,
                y: 0,
                width: this.CELL_SIZE,
                height: this.ROWS * this.CELL_SIZE,
                class: 'column-hover'
            });
            colGroup.appendChild(hoverArea);
        }

        for (let row = 0; row < this.ROWS; row++) {
            for (let col = 0; col < this.COLS; col++) {
                const cellCircle = this.createSvgElement('circle', {
                    cx: col * this.CELL_SIZE + this.CELL_SIZE / 2,
                    cy: row * this.CELL_SIZE + this.CELL_SIZE / 2,
                    r: this.CELL_SIZE / 2 - 5,
                    fill: this.COLORS[this.EMPTY],
                    'data-cell': `${row}-${col}`
                });
                columnGroups[col].appendChild(cellCircle);
            }
        }
    }

    createSvgElement(type, attributes = {}) {
        const element = document.createElementNS('http://www.w3.org/2000/svg', type);
        Object.entries(attributes).forEach(([key, value]) => {
            if (key === 'onclick') {
                element.onclick = value;
            } else {
                element.setAttribute(key, value);
            }
        });
        return element;
    }

    updateCellColors() {
        for (let row = 0; row < this.ROWS; row++) {
            for (let col = 0; col < this.COLS; col++) {
                const cell = this.boardSvg.querySelector(`[data-cell="${row}-${col}"]`);
                cell.setAttribute('fill', this.COLORS[this.board[row][col]]);
            }
        }
    }

    setValuation(value) {
        const offset = 50; // Centered at 50%
        const height = Math.abs(value) * 50; // Height is proportional to the absolute value

        // Apply smooth animation using CSS transitions
        this.valuationIndicator.style.transition = 'top 0.3s ease, height 0.3s ease';

        if (value > 0) {
            // Red (Player 1) grows upward
            this.valuationIndicator.style.top = `${offset - height}%`;
            this.valuationIndicator.style.height = `${height}%`;
            this.valuationIndicator.style.backgroundColor = this.COLORS[this.PLAYER1];
        } else {
            // Yellow (Player 2) grows downward
            this.valuationIndicator.style.top = `${offset}%`;
            this.valuationIndicator.style.height = `${height}%`;
            this.valuationIndicator.style.backgroundColor = this.COLORS[this.PLAYER2];
        }
    }

    onBoardStateChanged() {
        this.moveCounter.textContent = `Move ${this.currentMoveIndex} / ${this.moveHistory.length}`;

        this.backButton.disabled = this.firstButton.disabled = this.currentMoveIndex <= 0;
        this.forwardButton.disabled = this.lastButton.disabled = (this.currentMoveIndex >= this.moveHistory.length);

        if (this.gameOver) {
            if (this.isBoardFull())
                this.setValuation(0); // Draw
            else {
                console.log("Game Over: Player", this.getCurrentPlayer(-1), "wins!");
                this.setValuation(this.getCurrentPlayer(-1) === 1 ? 1 : -1); // Win
            }
        } else if (this.onnxModel) {
            const curPlayer = this.getCurrentPlayer();
            this.onnxModel.predict(this.board, curPlayer).then(output => {
                console.log("Model output:", output); // Debugging output
                const value = (curPlayer === this.PLAYER1) ? output.value : -output.value;
                this.setValuation(value);
            });
        }
    }

    checkForGameOver() {
        const lastPlayer = this.getCurrentPlayer(-1);
        const winningCells = this.checkWin(lastPlayer);
        if (winningCells) {
            this.gameOver = true;
            this.highlightWinningCells(winningCells);
            let statusText = `Player ${lastPlayer} wins!`;
            if (lastPlayer === this.humanPlayer)
                statusText += " 🎉";
            this.statusDiv.innerHTML = statusText;
            return lastPlayer;  // Player 1 or Player 2 wins
        } else if (this.isBoardFull()) {
            this.gameOver = true;
            this.statusDiv.innerHTML = "It's a draw!";
            return 3; // Draw
        } else {
            this.gameOver = false;
            const currentPlayer = this.getCurrentPlayer();
            const playerColor = this.COLORS[currentPlayer];
            const playerText = currentPlayer === this.humanPlayer
                ? "Your turn! Click on a column to drop a piece."
                : currentPlayer === this.cpuPlayer
                    ? "Computer's turn."
                    : `Player ${currentPlayer}'s turn.`;

            this.statusDiv.innerHTML = `
                <svg width="16" height="16" style="vertical-align: middle; margin-right: 5px;">
                    <circle cx="8" cy="8" r="8" fill="${playerColor}"></circle>
                </svg>
                ${playerText}
            `;
            return 0; // Not over
        }
    }

    handleColumnClick(col) {
        if (this.gameOver || this.getCurrentPlayer() !== this.humanPlayer) return;
        if (this.isColumnFull(col)) return;

        this.makeMove(col);
        this.checkIfComputerShouldMove();
    }

    checkIfComputerShouldMove() {
        if (!this.gameOver && this.getCurrentPlayer() === this.cpuPlayer) {
            this.statusDiv.textContent = "Computer is thinking...";
            setTimeout(() => this.makeComputerMove(this.getCurrentPlayer()), 500);
        }
    }

    getCurrentPlayer(offset = 0) {
        return (this.currentMoveIndex + offset) % 2 === 0 ? this.PLAYER1 : this.PLAYER2;
    }

    makeMove(col) {
        let row = this.ROWS - 1;
        while (row >= 0 && this.board[row][col] !== this.EMPTY) {
            row--;
        }

        if (row >= 0) {
            this.board[row][col] = this.getCurrentPlayer();
            this.moveHistory = this.moveHistory.slice(0, this.currentMoveIndex);
            this.moveHistory.push(col);
            this.currentMoveIndex++;
            this.updateCellColors();
            this.checkForGameOver();
            this.onBoardStateChanged();
            return true;
        }
        return false;
    }

    makeComputerMove(player) {
        if (this.onnxModel) {
            this.onnxModel.predict(this.board, player).then(output => {
                const sampledMove = output.sampledMove;
                this.makeMove(sampledMove);
            }).catch(error => {
                console.error("Error during computer move:", error);
            });
            return;
        }

        // Check if the computer can win in the next move
        const winningMove = this.findWinningMove(player);
        if (winningMove !== -1) {
            this.makeMove(winningMove);
            return;
        }

        // Block the opponent's winning move if any exist
        const blockingMove = this.findWinningMove(3 - player);
        if (blockingMove !== -1) {
            this.makeMove(blockingMove);
            return;
        }

        // Otherwise, pick a random valid move
        const validColumns = [];
        for (let col = 0; col < this.COLS; col++) {
            if (!this.isColumnFull(col)) {
                validColumns.push(col);
            }
        }

        if (validColumns.length > 0) {
            const randomCol = validColumns[Math.floor(Math.random() * validColumns.length)];
            this.makeMove(randomCol);
        }
    }

    findWinningMove(player) {
        for (let col = 0; col < this.COLS; col++) {
            if (this.isColumnFull(col)) continue;

            let row = this.ROWS - 1;
            while (row >= 0 && this.board[row][col] !== this.EMPTY) {
                row--;
            }

            if (row >= 0) {
                this.board[row][col] = player;
                const isWinning = this.checkWin(player);
                this.board[row][col] = this.EMPTY;

                if (isWinning) {
                    return col;
                }
            }
        }
        return -1;
    }

    isColumnFull(col) {
        return this.board[0][col] !== this.EMPTY;
    }

    isBoardFull() {
        for (let col = 0; col < this.COLS; col++) {
            if (!this.isColumnFull(col)) {
                return false;
            }
        }
        return true;
    }

    checkWin(player) {
        // Check horizontal wins
        for (let row = 0; row < this.ROWS; row++) {
            for (let col = 0; col <= this.COLS - 4; col++) {
                if (this.board[row][col] === player && 
                    this.board[row][col+1] === player && 
                    this.board[row][col+2] === player && 
                    this.board[row][col+3] === player) {
                    return [[row, col], [row, col+1], [row, col+2], [row, col+3]];
                }
            }
        }

        // Check vertical wins
        for (let row = 0; row <= this.ROWS - 4; row++) {
            for (let col = 0; col < this.COLS; col++) {
                if (this.board[row][col] === player && 
                    this.board[row+1][col] === player && 
                    this.board[row+2][col] === player && 
                    this.board[row+3][col] === player) {
                    return [[row, col], [row+1, col], [row+2, col], [row+3, col]];
                }
            }
        }

        // Check diagonal wins (down-right)
        for (let row = 0; row <= this.ROWS - 4; row++) {
            for (let col = 0; col <= this.COLS - 4; col++) {
                if (this.board[row][col] === player && 
                    this.board[row+1][col+1] === player && 
                    this.board[row+2][col+2] === player && 
                    this.board[row+3][col+3] === player) {
                    return [[row, col], [row+1, col+1], [row+2, col+2], [row+3, col+3]];
                }
            }
        }

        // Check diagonal wins (up-right)
        for (let row = 3; row < this.ROWS; row++) {
            for (let col = 0; col <= this.COLS - 4; col++) {
                if (this.board[row][col] === player && 
                    this.board[row-1][col+1] === player && 
                    this.board[row-2][col+2] === player && 
                    this.board[row-3][col+3] === player) {
                    return [[row, col], [row-1, col+1], [row-2, col+2], [row-3, col+3]];
                }
            }
        }

        return null;
    }

    highlightWinningCells(cells) {
        cells.forEach(([row, col]) => {
            const cell = this.boardSvg.querySelector(`[data-cell="${row}-${col}"]`);
            cell.classList.add('winning-cell');
        });
    }

    navigateBack() {
        if (this.currentMoveIndex > 0) {
            this.setHistoryIndex(this.currentMoveIndex - 1);
        }
    }

    navigateForward() {
        if (this.currentMoveIndex < this.moveHistory.length) {
            this.setHistoryIndex(this.currentMoveIndex + 1);
        }
    }

    navigateFirst() {
        if (this.currentMoveIndex > 0) {
            this.setHistoryIndex(0);
        }
    }

    navigateLast() {
        if (this.currentMoveIndex < this.moveHistory.length) {
            this.setHistoryIndex(this.moveHistory.length);
        }
    }

    setHistoryIndex(moveIndex) {
        this.currentMoveIndex = moveIndex;
        this.board = Array(this.ROWS).fill().map(() => Array(this.COLS).fill(this.EMPTY));

        for (let i = 0; i < moveIndex; i++) {
            const col = this.moveHistory[i];
            let row = this.ROWS - 1;
            while (row >= 0 && this.board[row][col] !== this.EMPTY) {
                row--;
            }
            if (row >= 0) {
                this.board[row][col] = (i % 2 === 0) ? this.PLAYER1 : this.PLAYER2;
            }
        }

        this.clearWinningOutlines();
        this.updateCellColors();
        this.checkForGameOver();
        this.onBoardStateChanged();
    }

    clearWinningOutlines() {
        this.boardSvg.querySelectorAll('.winning-cell').forEach(cell => {
            cell.classList.remove('winning-cell');
        });
    }
}


class Connect4ONNXModel {
    constructor(modelUrl) {
        this.modelUrl = modelUrl;
        this.session = null;
    }

    ready() {
        return this.session !== null;
    }

    async initialize() {
        try {
            console.log(`Loading ONNX model from: ${this.modelUrl}`);
            this.session = await ort.InferenceSession.create(this.modelUrl);
            console.log('ONNX model loaded successfully.');
        } catch (error) {
            console.error('Failed to load ONNX model:', error);
            throw error;
        }
    }

    async predict(boardState, curPlayer) {
        if (!this.session) {
            throw new Error('Model is not initialized. Call initialize() first.');
        }

        try {
            // Prepare the input tensor
            const formattedBoardState = boardState.map(row =>
                row.map(cell => cell === 0 ? 0 : (cell === curPlayer ? 1 : -1))
            );
            const inputTensor = new ort.Tensor('int8', new Int8Array(formattedBoardState.flat()), [6, 7]);

            // Run the model
            const feeds = { board: inputTensor };
            const results = await this.session.run(feeds);

            let logits = results.logits.data;
            const value = results.value.data[0];

            // Check the first row for invalid moves and mask them out
            for (let col = 0; col < boardState[0].length; col++) {
                if (boardState[0][col] !== 0) {
                    logits[col] = -Infinity;
                }
            }

            // Apply softmax to logits to get probabilities
            const maxLogit = Math.max(...logits);
            const expLogits = logits.map(logit => Math.exp(logit - maxLogit));
            const sumExpLogits = expLogits.reduce((sum, expLogit) => sum + expLogit, 0);
            const probabilities = expLogits.map(expLogit => expLogit / sumExpLogits);

            // Sample a move from the probability distribution
            const random = Math.random();
            let cumulativeProbability = 0;
            let sampledMove = -1;
            for (let i = 0; i < probabilities.length; i++) {
                cumulativeProbability += probabilities[i];
                if (random < cumulativeProbability) {
                    sampledMove = i;
                    break;
                }
            }

            return { logits, probabilities, sampledMove, value };
        } catch (error) {
            console.error('Failed to run prediction:', error);
            throw error;
        }
    }
}


// Initialize the game when the page loads
window.addEventListener('load', () => {
    document.querySelectorAll('.connect4-container').forEach(container => {
        const gameInstance = new Connect4Game(container);
    });
});