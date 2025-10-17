import { useState, useEffect, useCallback } from 'react';

const GRID_SIZE = 20;
const INITIAL_SPEED = 200; // 中等速度

export const useGameLogic = () => {
  const [snake, setSnake] = useState([{ x: 10, y: 10 }]);
  const [food, setFood] = useState({ x: 15, y: 15 });
  const [direction, setDirection] = useState('right');
  const [gameRunning, setGameRunning] = useState(false);
  const [gamePaused, setGamePaused] = useState(false);
  const [gameOver, setGameOver] = useState(false);
  const [score, setScore] = useState(0);
  const [highScore, setHighScore] = useState(() => {
    const saved = localStorage.getItem('snakeHighScore');
    return saved ? parseInt(saved) : 0;
  });

  const generateFood = useCallback((currentSnake) => {
    let newFood;
    do {
      newFood = {
        x: Math.floor(Math.random() * GRID_SIZE),
        y: Math.floor(Math.random() * GRID_SIZE)
      };
    } while (currentSnake.some(segment => segment.x === newFood.x && segment.y === newFood.y));
    return newFood;
  }, []);

  const checkCollision = useCallback((head, currentSnake) => {
    // 检查墙壁碰撞
    if (head.x < 0 || head.x >= GRID_SIZE || head.y < 0 || head.y >= GRID_SIZE) {
      return true;
    }
    
    // 检查自身碰撞
    return currentSnake.some(segment => segment.x === head.x && segment.y === head.y);
  }, []);

  const moveSnake = useCallback(() => {
    if (!gameRunning || gamePaused || gameOver) return;

    setSnake(currentSnake => {
      const newSnake = [...currentSnake];
      const head = { ...newSnake[0] };

      // 根据方向移动头部
      switch (direction) {
        case 'up':
          head.y -= 1;
          break;
        case 'down':
          head.y += 1;
          break;
        case 'left':
          head.x -= 1;
          break;
        case 'right':
          head.x += 1;
          break;
        default:
          break;
      }

      // 检查碰撞
      if (checkCollision(head, newSnake)) {
        setGameOver(true);
        setGameRunning(false);
        return currentSnake;
      }

      newSnake.unshift(head);

      // 检查是否吃到食物
      if (head.x === food.x && head.y === food.y) {
        setScore(prev => {
          const newScore = prev + 10;
          if (newScore > highScore) {
            setHighScore(newScore);
            localStorage.setItem('snakeHighScore', newScore.toString());
          }
          return newScore;
        });
        setFood(generateFood(newSnake));
      } else {
        newSnake.pop();
      }

      return newSnake;
    });
  }, [direction, food, gameRunning, gamePaused, gameOver, checkCollision, generateFood, highScore]);

  useEffect(() => {
    if (!gameRunning || gamePaused || gameOver) return;

    const gameInterval = setInterval(moveSnake, INITIAL_SPEED);
    return () => clearInterval(gameInterval);
  }, [moveSnake, gameRunning, gamePaused, gameOver]);

  const startGame = () => {
    setGameRunning(true);
    setGamePaused(false);
    setGameOver(false);
  };

  const pauseGame = () => {
    setGamePaused(true);
  };

  const resumeGame = () => {
    setGamePaused(false);
  };

  const resetGame = () => {
    setSnake([{ x: 10, y: 10 }]);
    setFood({ x: 15, y: 15 });
    setDirection('right');
    setScore(0);
    setGameRunning(false);
    setGamePaused(false);
    setGameOver(false);
  };

  return {
    snake,
    food,
    score,
    highScore,
    gameOver,
    direction,
    setDirection,
    startGame,
    pauseGame,
    resumeGame,
    resetGame
  };
};
