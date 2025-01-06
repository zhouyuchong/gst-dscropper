/*
 * @Author: zhouyuchong
 * @Date: 2024-12-30 16:10:31
 * @Description: 
 * @LastEditors: zhouyuchong
 * @LastEditTime: 2024-12-31 11:00:12
 */
#ifndef MY_FDFS_CLIENT_H
#define MY_FDFS_CLIENT_H
 
#include "fdfs_client.h"
#include "fdfs_global.h"
#include "base64.h"
#include "sockopt.h"
#include "logger.h"
#include <string>
#define TEST
using namespace std;
 
//错误码
enum FSC_CODE
{
    FSC_ERROR_CODE_NONE = 0,				//没有错误
    FSC_ERROR_CODE_INIT_FAILED,				//初始化失败
 
    FSC_ERROR_CODE_PARAM_INVAILD,			//参数无效
 
    FSC_ERROR_CODE_CONNECT_TRACKER_FAIL,    //连接失败
    FSC_ERROR_CODE_QUERY_STORAGE_FAIL,		//查询storage地址失败
    FSC_ERROR_CODE_CONNECT_STORAGE_FAIL,	//连接storage失败
 
    FSC_ERROR_CODE_DOWNLAOD_FILE_FAIL,		//下载文件失败
    FSC_ERROR_CODE_DELETE_FILE_FAIL,		//删除文件失败
};
 
class CFDFSClient
{
public:
	CFDFSClient(void);
	~CFDFSClient(void);
 
public:
	
	// 初始化客户端
    //
    //功能：初始化fastdfs
    //参数：
    //      const char* sConfig IN FastDFS配置文件路劲 比如:/etc/fdfs/client.conf
    //		int nLogLevel 日志等级 采用的是unix 日志等级
    //  0: LOG_EMERG
    //	1: LOG_ALERT
    //	2: LOG_CRIT
    //	3: LOG_ERR
    //	4: LOG_WARNING
    //	5: LOG_NOTICE
    //	6: LOG_INFO
    //	7: LOG_DEBUG
 
    //返回：int& anError OUT 错误信息
 
    //备注：
    //      注意初始化时，必须保证conf文件中base_path目录存在
    //		比如 base_path=/fastdfs/tracker, 需要保证/fastdfs/tracker存在，
    //		不存在 需创建mkdir /fastdfs/tracker
    //
	int init(const char* sFDFSConfig, int nLogLevel);	
	// 上传
    //
    //功能：上传文件
    //参数：
    //      char *file_content IN 文件内容
    //      const char *file_ext_name IN 文件扩展名
    //		int file_size IN 文件大小
    //		int& name_size	OUT 返回的文件名大小
    //      char* remote_file_name OUT 返回的文件名
    //      比如：group2/M00/00/00/CgEIzVRhnJWAZfVkAAAsFwWtoVg250.png
 
    // 返回：0:成功 否则失败。
    //
	int fdfs_uploadfile( const char *file_content, const char *file_ext_name, int file_size, 
        int& name_size, char* &remote_file_name);
   
	// 所有组信息
    //
    //功能：获取所有组信息
    //参数：
    //      BufferInfo* group_info OUT 所有组信息
 
    // 返回：0:成功 否则失败。
    //
 
private:
	void re_fastfds_client_init();
 
	int fastfdfs_client_init(const char* sFDFSConfig);
 
private:
	ConnectionInfo *m_pTrackerServer;
	BufferInfo m_RecvBufferInfo;
	char* m_pRemoteFileName;
	string m_strConfigPath;
	int	m_nLevelLog;
};
 
#endif
 