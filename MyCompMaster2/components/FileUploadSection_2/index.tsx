import React from '@alipay/bigfish/react';
import { Button } from '@alipay/bigfish/antd';
import { StarOutlined, ClockCircleOutlined } from '@ant-design/icons';

import styles from './index.less';

const FileUploadSection_2: React.FC = () => {
  const onButtonClick = () => {};

  return (
    <div className={styles.wrapper3}>
      <div className={styles.clsDiv4}>
        <span className={styles.clsSpan2}>Hi!你好</span>
        <span className={styles.clsSpan3}>你好</span>
        <Button
          shape="round"
          size="large"
          onClick={onButtonClick}
          className={styles.clsButton2}
        >
          开店
        </Button>
        <div className={styles.clsDiv5}>
          <StarOutlined className={styles.clsIcon2} />
          <div className={styles.clsDiv6}>
            <div className={styles.clsDiv7}>
              <span className={styles.componentCommonLzd9}>宝贝收藏</span>
              <span className={styles.componentSharedIst5}>买过的店</span>
              <span className={styles.componentSharedIst5}>收藏的店</span>
            </div>
            <div className={styles.clsDiv8}>
              <span className={styles.componentCommonGvh9}>Hi!</span>
              <span className={styles.componentCommonGvh9}>你好</span>
            </div>
          </div>
          <ClockCircleOutlined className={styles.clsIcon3} />
        </div>
      </div>
      <div className={styles.clsDiv9}>
        <span className={styles.componentSharedIst5}>宝贝收藏</span>
        <span className={styles.componentSharedIst5}>买过的店</span>
        <span className={styles.componentSharedIst5}>收藏的店</span>
        <span className={styles.componentSharedIst5}>我的足迹</span>
      </div>
    </div>
  );
};

export default FileUploadSection_2;
