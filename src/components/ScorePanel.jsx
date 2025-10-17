import React from 'react';
import { Statistic, Row, Col } from 'antd';
import { TrophyOutlined, StarOutlined } from '@ant-design/icons';
import './ScorePanel.less';

const ScorePanel = ({ score, highScore }) => {
  return (
    <div className="score-panel">
      <Row gutter={16} justify="center">
        <Col>
          <Statistic
            title="当前分数"
            value={score}
            prefix={<StarOutlined />}
            valueStyle={{ color: '#1890ff' }}
          />
        </Col>
        <Col>
          <Statistic
            title="最高分数"
            value={highScore}
            prefix={<TrophyOutlined />}
            valueStyle={{ color: '#faad14' }}
          />
        </Col>
      </Row>
    </div>
  );
};

export default ScorePanel;
