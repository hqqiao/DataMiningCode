import React from '@alipay/bigfish/react';
import { StarOutlined, ClockCircleOutlined } from '@ant-design/icons';
import { Button } from '@alipay/bigfish/antd';

import styles from './index.less';

const FileUploadSection_3: React.FC = () => {
  const onButtonClick = () => {};

  const onButton2Click = () => {};

  return (
    <div className={styles.wrapper4}>
      <div className={styles.clsDiv11}>
        <div className={styles.clsDiv12}>
          <span className={styles.componentCommonZsr2}>Hi!你好</span>
          <span className={styles.componentCommonZsr2}>Hi!你好</span>
        </div>
        <div className={styles.uploadArea}>
          <StarOutlined className={styles.clsIcon4} />
          <div className={styles.clsDiv14}>
            <Button
              shape="round"
              size="large"
              onClick={onButtonClick}
              className={styles.clsButton3}
            >
              <span className={styles.clsSpan15}>开店</span>
              <span className={styles.clsSpan16}>Hi!</span>
            </Button>
            <ClockCircleOutlined className={styles.clsIcon5} />
            <Button
              shape="round"
              size="large"
              onClick={onButton2Click}
              className={styles.clsButton4}
            >
              开店
            </Button>
          </div>
        </div>
      </div>
      <div className={styles.clsDiv15}>
        <div className={styles.clsDiv16}>
          <div className={styles.clsDiv17}>
            <span className={styles.componentSharedIst5}>宝贝收藏</span>
            <span className={styles.componentSharedIst5}>买过的店</span>
            <span className={styles.componentSharedIst5}>收藏的店</span>
            <span className={styles.componentSharedIst5}>我的足迹</span>
          </div>
          <ClockCircleOutlined className={styles.clsIcon6} />
          <StarOutlined className={styles.clsIcon7} />
        </div>
        <StarOutlined className={styles.clsIcon8} />
        <div className={styles.clsDiv18}>
          <div className={styles.clsDiv19}>
            <span className={styles.componentSharedIst5}>宝贝收藏</span>
            <span className={styles.componentSharedIst5}>买过的店</span>
            <span className={styles.componentSharedIst5}>收藏的店</span>
          </div>
          <span className={styles.componentSharedIst5}>我的足迹</span>
        </div>
      </div>
    </div>
  );
};

export default FileUploadSection_3;
