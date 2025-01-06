#include "cfdfs_client.h"
#ifndef TEST
// #include "json/json.h"
#endif
 
#define  MAX_REMOTE_FILE_NAME_SIZE 100
 
CFDFSClient::CFDFSClient(void)
{
	m_pTrackerServer = NULL;
	//m_RecvBufferInfo = {0};
	memset(&m_RecvBufferInfo,0,sizeof(m_RecvBufferInfo));
	m_pRemoteFileName = NULL;
}
 
CFDFSClient::~CFDFSClient(void)
{
	fdfs_client_destroy();
 
	log_destroy();
	
	if (m_RecvBufferInfo.buff != NULL)
	{
		free(m_RecvBufferInfo.buff);
	}
 
	if (m_pRemoteFileName != NULL)
	{
		free(m_pRemoteFileName);
	}
}
 
int CFDFSClient::init( const char* sFDFSConfig, int nLogLevel)
{
	// g_log_context
    log_init(); 
    g_log_context.log_level = nLogLevel;
 
	m_strConfigPath = sFDFSConfig;
	m_nLevelLog = nLogLevel;
 
	// 初始化fastfds客户端
	int result = 0;
	result = fastfdfs_client_init(sFDFSConfig);
 
    m_pRemoteFileName = (char*)malloc(MAX_REMOTE_FILE_NAME_SIZE * sizeof(char));
	memset(m_pRemoteFileName, 0, MAX_REMOTE_FILE_NAME_SIZE - 1);
	return result;
}
 
int CFDFSClient::fastfdfs_client_init(const char* sFDFSConfig)
{
	int result = 0;
	if ((result=fdfs_client_init(m_strConfigPath.c_str())) != 0)
	{
		logErrorEx(&g_log_context, "CFDFSClient::init() fdfs_client_init is failed, result:%d", result);
		return FSC_ERROR_CODE_INIT_FAILED;
	}
 
	return result;
}
 
int CFDFSClient::fdfs_uploadfile(const char *file_content, const char *file_ext_name, int file_size,
                                 int& name_size, char *&remote_file_name)
{
    int result = 0;
    ConnectionInfo *pTrackerServer = tracker_get_connection();
	if (pTrackerServer == NULL)
	{
		result = (errno != 0 ? errno : ECONNREFUSED);
		logErrorEx(&g_log_context, "CFDFSClient::fdfs_uploadfile() tracker_get_connection is failed, result:%d", result);
 
		return FSC_ERROR_CODE_CONNECT_TRACKER_FAIL;
	}
    
	char group_name[FDFS_GROUP_NAME_MAX_LEN + 1];
	char remote_filename[256];
 
	int store_path_index;
	ConnectionInfo storageServer;
	ConnectionInfo* pStorageServer;
	if ((result=tracker_query_storage_store(pTrackerServer, \
		&storageServer, group_name, &store_path_index)) != 0)
	{
        tracker_disconnect_server_ex(pTrackerServer, true);
 
		logErrorEx(&g_log_context, "tracker_query_storage fail, " \
			"error no: %d, error info: %s\n", \
			result, STRERROR(result));
		return result;
	}
 
	if ((pStorageServer=tracker_connect_server(&storageServer, \
		&result)) == NULL)
	{
		logErrorEx(&g_log_context, "CFDFSClient::fdfs_uploadfile() \
								   tracker_connect_server failed, result:%d, storage=%s:%d\n", 
								   result, storageServer.ip_addr, \
								   storageServer.port);
        tracker_disconnect_server_ex(pTrackerServer, true);
 
		return result;
	}
 
	result = storage_upload_by_filebuff(pTrackerServer, \
		pStorageServer, store_path_index, \
		file_content, file_size, file_ext_name, \
		NULL, 0, \
		group_name, remote_filename);
	if (result != 0)
	{
		const char* strMsg = STRERROR(result);
		logErrorEx(&g_log_context, "CFDFSClient::fdfs_uploadfile() upload file fail, " \
			"group:%s, remote;%s, error no: %d, error info: %s\n", \
			group_name, remote_filename, result, strMsg);
	}
    else{
        int nNameSize = snprintf(m_pRemoteFileName,MAX_REMOTE_FILE_NAME_SIZE-1, "%s/%s", group_name, remote_filename);
        remote_file_name = m_pRemoteFileName;
        name_size = nNameSize;
    }
    
    tracker_disconnect_server_ex(pStorageServer, true);
    tracker_disconnect_server_ex(pTrackerServer, true);
 
    return result;
}
 
