import React from '@alipay/bigfish/react';
import FileUploadSection from './components/FileUploadSection';
import FileUploadSection_2 from './components/FileUploadSection_2';
import FileUploadSection_3 from './components/FileUploadSection_3';
import FileUploadSection_4 from './components/FileUploadSection_4';
import FileUploadSection_5 from './components/FileUploadSection_5';
import FileUploadSection_6 from './components/FileUploadSection_6';

import styles from './index.less';

const MyPageComponent: React.FC = () => {
  return (
    <div className={styles.wrapper}>
      <div className={styles.uploadArea}>
        <div className={styles.clsDiv2}>
          <div className={styles.clsDiv3}>
            <FileUploadSection />
            <FileUploadSection_2 />
            <div className={styles.clsDiv10}>
              <FileUploadSection_3 />
              <FileUploadSection_4 />
            </div>
          </div>
          <FileUploadSection_5 />
        </div>
        <FileUploadSection_6 />
      </div>
    </div>
  );
};

export default MyPageComponent;
