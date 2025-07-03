import React from '@alipay/bigfish/react';
import { Button } from '@alipay/bigfish/antd';
import { ClockCircleOutlined } from '@ant-design/icons';

import styles from './index.less';

const FileUploadSection_6: React.FC = () => {
  const onButtonClick = () => {};

  return (
    <div className={styles.wrapper7}>
      <span className={styles.clsSpan29}>Hi!你好</span>
      <Button
        shape="round"
        size="large"
        onClick={onButtonClick}
        className={styles.clsButton6}
      >
        开店
      </Button>
      <ClockCircleOutlined className={styles.clsIcon12} />
      <div className={styles.clsDiv24}>
        <span className={styles.componentSharedIst5}>买过的店</span>
        <span className={styles.componentSharedIst5}>收藏的店</span>
        <span className={styles.componentSharedIst5}>我的足迹</span>
      </div>
    </div>
  );
};

export default FileUploadSection_6;
