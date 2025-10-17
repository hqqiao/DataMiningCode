import React from 'react';
import { Modal, Button, Typography, Result } from 'antd';
import { TrophyOutlined, ReloadOutlined } from '@ant-design/icons';
import './GameOverModal.less';

const { Title, Text } = Typography;

const GameOverModal = ({ visible, score, highScore, onRestart, onClose }) => {
  const isNewRecord = score === highScore && score > 0;

  return (
    <Modal
      title="游戏结束"
      open={visible}
      onCancel={onClose}
      footer={null}
      centered
      className="game-over-modal"
      width={400}
    >
      <Result
        status={isNewRecord ? "success" : "info"}
        title={isNewRecord ? "🎉 新纪录！" : "游戏结束"}
        subTitle={
          <div className="result-content">
            <Text strong>最终得分：{score}</Text>
            <br />
            <Text type="secondary">最高纪录：{highScore}</Text>
            {isNewRecord && (
              <div className="new-record-badge">
                <TrophyOutlined /> 新纪录！
              </div>
            )}
          </div>
        }
      />
      <div className="modal-actions">
        <Button
          type="primary"
          icon={<ReloadOutlined />}
          onClick={onRestart}
          size="large"
          block
        >
          再玩一次
        </Button>
      </div>
    </Modal>
  );
};

export default GameOverModal;
