import React from 'react';
import { Empty } from 'antd';
import './GameBoard.less';

const GameBoard = ({ snake, food, gameStarted, gamePaused, gameOver }) => {
  const GRID_SIZE = 20;
  const BOARD_SIZE = 400;

  const renderGrid = () => {
    const grid = [];
    for (let row = 0; row < GRID_SIZE; row++) {
      for (let col = 0; col < GRID_SIZE; col++) {
        const isSnake = snake.some(segment => segment.x === col && segment.y === row);
        const isFood = food.x === col && food.y === row;
        const isHead = snake[0] && snake[0].x === col && snake[0].y === row;

        grid.push(
          <div
            key={`${row}-${col}`}
            className={`grid-cell ${isSnake ? 'snake' : ''} ${isHead ? 'snake-head' : ''} ${isFood ? 'food' : ''}`}
          />
        );
      }
    }
    return grid;
  };

  return (
    <div className="game-board">
      <div 
        className="game-grid"
        style={{ width: BOARD_SIZE, height: BOARD_SIZE }}
      >
        {gameStarted ? (
          renderGrid()
        ) : (
          <div className="game-placeholder">
            <Empty 
              description="点击开始按钮开始游戏"
              imageStyle={{ height: 100 }}
            />
          </div>
        )}
      </div>
      
      {gamePaused && gameStarted && !gameOver && (
        <div className="game-status-overlay">
          <div className="status-text">游戏暂停</div>
        </div>
      )}
      
      {gameOver && (
        <div className="game-status-overlay game-over">
          <div className="status-text">游戏结束</div>
        </div>
      )}
    </div>
  );
};

export default GameBoard;
