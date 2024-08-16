#pragma once

#include <ns3/applications-module.h>

class NetApp : public ns3::Application {
private:
  void StartApplication() override { NS_LOG_INFO("APP started"); }
  void StopApplication() override { NS_LOG_INFO("APP stopped"); }
};