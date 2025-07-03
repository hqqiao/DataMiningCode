import React from '@alipay/bigfish/react';
import { Button } from '@alipay/bigfish/antd';
import { StarOutlined, ClockCircleOutlined } from '@ant-design/icons';

import styles from './index.less';

const FileUploadSection_5: React.FC = () => {
  const onButtonClick = () => {};

  return (
    <div className={styles.wrapper6}>
      <div className={styles.clsDiv20}>
        <Button
          shape="round"
          size="large"
          onClick={onButtonClick}
          className={styles.clsButton5}
        >
          开店
        </Button>
        <div className={styles.clsDiv21}>
          <StarOutlined className={styles.componentSharedHjt7} />
          <div className={styles.clsDiv22}>
            <ClockCircleOutlined className={styles.componentSharedHjt7} />
            <StarOutlined className={styles.componentSharedHjt7} />
          </div>
        </div>
      </div>
      <div className={styles.clsDiv23}>
        <span className={styles.componentSharedYmm9}>宝贝收藏</span>
        <span className={styles.componentSharedYmm9}>买过的店</span>
        <span className={styles.componentSharedYmm9}>收藏的店</span>
        <span className={styles.componentSharedYmm9}>我的足迹宝贝收藏</span>
      </div>
    </div>
  );
};

export default FileUploadSection_5;
