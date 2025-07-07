import React, { useState } from 'react';
import {
  TabletOutlined,
  CheckOutlined,
  QuestionCircleOutlined,
} from '@ant-design/icons';
import { Switch, Divider, Button } from 'antd';

import styles from './index.less';

const ThreeDSSettingsPanel: React.FC = () => {
  const [smsNotifySwitch, setSmsNotifySwitch] = useState(false);
  const [worldCardNotifySwitch, setWorldCardNotifySwitch] = useState(true);

  const onEditClick = () => {};

  return (
    <div className={styles.wrapper}>
      <span className={styles.threeDsVerificationTitle}>3DS验证码设置</span>
      <div className={styles.settingsContainer}>
        <div className={styles.notificationChannelsSection}>
          <span className={styles.channelLabel}>通知渠道</span>
          <div className={styles.channelItem}>
            <div className={styles.componentCommonBqr4}>
              <div className={styles.componentSharedSzg0}>
                <div className={styles.iconWrapper}>
                  <TabletOutlined className={styles.tabletIcon} />
                </div>
                <div className={styles.componentSharedPpy7}>
                  <span className={styles.componentSharedMiu6}>短信通知</span>
                  <span className={styles.componentSharedHac1}>
                    3DS交易验证短信通知默认开启，短信将会发送至您卡片设置的3DS手机号
                  </span>
                </div>
              </div>
              <Switch
                checked={smsNotifySwitch}
                onChange={e => {
                  setSmsNotifySwitch(e);
                }}
              />
            </div>
            <div className={styles.componentCommonBqr4}>
              <div className={styles.componentSharedSzg0}>
                <div className={styles.imageWrapper}>
                  <img
                    alt=""
                    src="https://mdn.alipayobjects.com/fecodex_image/afts/img/7iwaT7n-aUUAAAAAG8AAAAgAejH3AQBr/original"
                    className={styles.platformNotificationImage}
                  />
                </div>
                <div className={styles.componentSharedPpy7}>
                  <span className={styles.componentSharedMiu6}>
                    World Card 平台内通知
                  </span>
                  <span className={styles.componentSharedHac1}>
                    3DS 交易验证World Card 平台内通知，可在 World Card 3DS
                    验证通知查看
                  </span>
                </div>
              </div>
              <Switch
                checked={worldCardNotifySwitch}
                onChange={e => {
                  setWorldCardNotifySwitch(e);
                }}
              />
            </div>
          </div>
        </div>
        <Divider className={styles.sectionDivider} />
        <div className={styles.phoneNumberSection}>
          <div className={styles.phoneNumberContent}>
            <div className={styles.componentSharedCms8}>
              <span className={styles.componentSharedEjz8}>通知接收手机号</span>
              <div className={styles.selectedPhoneNumber}>
                <CheckOutlined className={styles.componentSharedJyo0} />
                <span className={styles.componentSharedJyl8}>卡3DS手机号</span>
              </div>
            </div>
            <QuestionCircleOutlined className={styles.questionIcon} />
          </div>
          <Divider className={styles.subSectionDivider} />
          <div className={styles.updatePhoneEmailSection}>
            <div className={styles.componentSharedCms8}>
              <span className={styles.componentSharedEjz8}>
                更新账户安全手机号/邮箱，是否覆盖卡3DS手机号/邮箱？
              </span>
              <div className={styles.coverageOption}>
                <CheckOutlined className={styles.componentSharedJyo0} />
                <span className={styles.componentSharedJyl8}>覆盖</span>
              </div>
            </div>
            <Button
              shape="round"
              onClick={onEditClick}
              className={styles.editButton}
            >
              修改
            </Button>
          </div>
        </div>
      </div>
    </div>
  );
};

export default ThreeDSSettingsPanel;
