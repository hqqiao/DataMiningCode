import React, { useState, useEffect, useCallback } from 'react';
import { Card, Button, Typography, Space, Row, Col } from 'antd';
import { PlayCircleOutlined, PauseCircleOutlined, ReloadOutlined } from '@ant-design/icons';
import GameBoard from './GameBoard';
import ScorePanel from './ScorePanel';
import GameOverModal from './GameOverModal';
import { useGameLogic } from '../hooks/useGameLogic';
import './GameContainer.less';

const { Title } = Typography;

const GameContainer = () => {
  const [gameStarted, setGameStarted] = useState(false);
  const [gamePaused, setGamePaused] = useState(false);
  const [showGameOver, setShowGameOver] = useState(false);
  
  const {
    snake,
    food,
    score,
    highScore,
    gameOver,
    startGame,
    pauseGame,
    resumeGame,
    resetGame,
    direction,
    setDirection
  } = useGameLogic();

  useEffect(() => {
    if (gameOver) {
      setShowGameOver(true);
      setGameStarted(false);
      setGamePaused(false);
    }
  }, [gameOver]);

  const handleStart = () => {
    if (!gameStarted) {
      startGame();
      setGameStarted(true);
      setGamePaused(false);
    } else if (gamePaused) {
      resumeGame();
      setGamePaused(false);
    } else {
      pauseGame();
      setGamePaused(true);
    }
  };

  const handleReset = () => {
    resetGame();
    setGameStarted(false);
    setGamePaused(false);
    setShowGameOver(false);
  };

  const handleKeyPress = useCallback((e) => {
    if (!gameStarted || gamePaused || gameOver) return;
    
    const key = e.key.toLowerCase();
    const newDirection = direction;
    
    switch (key) {
      case 'arrowup':
      case 'w':
        if (direction !== 'down') setDirection('up');
        break;
      case 'arrowdown':
      case 's':
        if (direction !== 'up') setDirection('down');
        break;
      case 'arrowleft':
      case 'a':
        if (direction !== 'right') setDirection('left');
        break;
      case 'arrowright':
      case 'd':
        if (direction !== 'left') setDirection('right');
        break;
      default:
        break;
    }
  }, [gameStarted, gamePaused, gameOver, direction, setDirection]);

  useEffect(() => {
    window.addEventListener('keydown', handleKeyPress);
    return () => window.removeEventListener('keydown', handleKeyPress);
  }, [handleKeyPress]);

  return (
    <div className="game-container">
      <Card className="game-card">
        <div className="game-header">
          <Title level={2} className="game-title">贪吃蛇游戏</Title>
          <ScorePanel score={score} highScore={highScore} />
        </div>
        
        <GameBoard 
          snake={snake} 
          food={food} 
          gameStarted={gameStarted}
          gamePaused={gamePaused}
          gameOver={gameOver}
        />
        
        <div className="game-controls">
          <Row gutter={16} justify="center">
            <Col>
              <Button
                type="primary"
                size="large"
                icon={gameStarted && !gamePaused ? <PauseCircleOutlined /> : <PlayCircleOutlined />}
                onClick={handleStart}
                disabled={gameOver}
              >
                {gameStarted && !gamePaused ? '暂停' : '开始'}
              </Button>
            </Col>
            <Col>
              <Button
                size="large"
                icon={<ReloadOutlined />}
                onClick={handleReset}
              >
                重新开始
              </Button>
            </Col>
          </Row>
        </div>
        
        <div className="game-instructions">
          <p>使用方向键或 WASD 键控制蛇的移动</p>
        </div>
      </Card>
      
      <GameOverModal 
        visible={showGameOver} 
        score={score} 
        highScore={highScore}
        onRestart={handleReset}
        onClose={() => setShowGameOver(false)}
      />
    </div>
  );
};

export default GameContainer;
