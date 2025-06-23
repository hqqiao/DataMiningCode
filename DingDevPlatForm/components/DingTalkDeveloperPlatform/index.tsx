import React from 'react';
import { Button, Divider } from 'antd';

import styles from './index.less';

const DingTalkDeveloperPlatform: React.FC = () => {
  const onGoNowClick = () => {};

  const onGoNowClick2 = () => {};

  return (
    <div className={styles.wrapper}>
      <div className={styles.platformHeader}>
        <img
          alt=""
          src="https://weavefox.alipay.com/assets/cce3ef5a-cb57-4266-8377-58a22bdea1e0.png"
          className={styles.logoImage}
        />
        <a className={styles.platformLink}>钉钉开放平台</a>
      </div>
      <span className={styles.platformDescription}>
        集团开发者请根据开发类型选择相应平台
      </span>
      <div className={styles.platformOptions}>
        <div className={styles.internalAppSection}>
          <span className={styles.componentSharedYdo6}>开发阿里巴巴内部应用</span>
          <span className={styles.internalAppText}>
            开发仅限集团内部使用的应用，请移步
            <a className={styles.componentSharedXmy3}>轻研·阿里钉开放平台</a>
          </span>
          <Button
            type="primary"
            onClick={onGoNowClick}
            className={styles.internalAppButton}
          >
            立即前往
          </Button>
        </div>
        <div className={styles.thirdPartyAppSection}>
          <div className={styles.thirdPartyAppTitle}>
            <span className={styles.componentSharedYdo6}>开发第三方企业应用</span>
            <div className={styles.thirdPartyAppDescription}>
              <span className={styles.thirdPartyAppText}>
                开发上架到钉钉应用市场的应用，可在开发者后台选择其他组织开发
              </span>
              <div className={styles.thirdPartyAppActions}>
                <Button
                  type="primary"
                  onClick={onGoNowClick2}
                  className={styles.thirdPartyAppButton}
                >
                  立即前往
                </Button>
                <a className={styles.whatIsThirdPartyAppLink}>
                  什么是三方应用?
                </a>
              </div>
            </div>
          </div>
          <Divider className={styles.sectionDivider} />
          <div className={styles.additionalResources}>
            <div className={styles.resourceLinks}>
              <img
                alt=""
                src="https://weavefox.alipay.com/assets/84503dfc-1814-4425-b5f5-d9069de8f526.png"
                className={styles.componentCommonQay8}
              />
              <a className={styles.componentSharedLes5}>接入流程</a>
              <img
                alt=""
                src="https://weavefox.alipay.com/assets/e30a9eb3-543b-45c0-bd6e-9c6ebcb09571.png"
                className={styles.componentCommonQay8}
              />
              <a className={styles.componentSharedLes5}>开发文档</a>
              <img
                alt=""
                src="https://weavefox.alipay.com/assets/e3dc08f9-288c-431b-88b3-325292f39abc.png"
                className={styles.componentCommonQay8}
              />
            </div>
            <a className={styles.officialWebsiteLink}>钉钉开放平台官网</a>
            <span className={styles.serviceGroupInfo}>
              <a className={styles.componentSharedXmy3}>内部服务群:</a>
              33317273
            </span>
          </div>
        </div>
      </div>
    </div>
  );
};

export default DingTalkDeveloperPlatform;
