import React from '@alipay/bigfish/react';
import { Button } from '@alipay/bigfish/antd';
import { ClockCircleOutlined } from '@ant-design/icons';

import styles from './index.less';

const FileUploadSection: React.FC = () => {
  const onButtonClick = () => {};

  return (
    <div className={styles.wrapper2}>
      <Button
        shape="round"
        size="large"
        onClick={onButtonClick}
        className={styles.clsButton}
      >
        开店
      </Button>
      <ClockCircleOutlined className={styles.clsIcon} />
      <span className={styles.clsSpan}>我的足迹</span>
    </div>
  );
};

export default FileUploadSection;
