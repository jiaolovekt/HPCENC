/*** Autogenerated by WIDL 8.0.1 from ../include/dxgidebug.idl - Do not edit ***/

#ifdef _WIN32
#ifndef __REQUIRED_RPCNDR_H_VERSION__
#define __REQUIRED_RPCNDR_H_VERSION__ 475
#endif
#include <rpc.h>
#include <rpcndr.h>
#endif

#ifndef COM_NO_WINDOWS_H
#include <windows.h>
#include <ole2.h>
#endif

#ifndef __dxgidebug_h__
#define __dxgidebug_h__

#ifndef __WIDL_INLINE
#if defined(__cplusplus) || defined(_MSC_VER)
#define __WIDL_INLINE inline
#elif defined(__GNUC__)
#define __WIDL_INLINE __inline__
#endif
#endif

/* Forward declarations */

#ifndef __IDXGIInfoQueue_FWD_DEFINED__
#define __IDXGIInfoQueue_FWD_DEFINED__
typedef interface IDXGIInfoQueue IDXGIInfoQueue;
#ifdef __cplusplus
interface IDXGIInfoQueue;
#endif /* __cplusplus */
#endif

#ifndef __IDXGIDebug_FWD_DEFINED__
#define __IDXGIDebug_FWD_DEFINED__
typedef interface IDXGIDebug IDXGIDebug;
#ifdef __cplusplus
interface IDXGIDebug;
#endif /* __cplusplus */
#endif

#ifndef __IDXGIDebug1_FWD_DEFINED__
#define __IDXGIDebug1_FWD_DEFINED__
typedef interface IDXGIDebug1 IDXGIDebug1;
#ifdef __cplusplus
interface IDXGIDebug1;
#endif /* __cplusplus */
#endif

/* Headers for imported files */

#include <oaidl.h>

#ifdef __cplusplus
extern "C" {
#endif

#define DXGI_DEBUG_BINARY_VERSION (1)

typedef GUID DXGI_DEBUG_ID;
DEFINE_GUID(DXGI_DEBUG_ALL,   0xe48ae283, 0xda80, 0x490b,0x87, 0xe6, 0x43, 0xe9, 0xa9, 0xcf, 0xda, 0x08);
DEFINE_GUID(DXGI_DEBUG_DX,    0x35cdd7fc, 0x13b2, 0x421d,0xa5, 0xd7, 0x7e, 0x44, 0x51, 0x28, 0x7d, 0x64);
DEFINE_GUID(DXGI_DEBUG_DXGI,  0x25cddaa4, 0xb1c6, 0x47e1,0xac, 0x3e, 0x98, 0x87, 0x5b, 0x5a, 0x2e, 0x2a);
DEFINE_GUID(DXGI_DEBUG_APP,   0x06cd6e01, 0x4219, 0x4ebd,0x87, 0x90, 0x27, 0xed, 0x23, 0x36, 0x0c, 0x62);
typedef enum DXGI_DEBUG_RLO_FLAGS {
    DXGI_DEBUG_RLO_SUMMARY = 0x1,
    DXGI_DEBUG_RLO_DETAIL = 0x2,
    DXGI_DEBUG_RLO_IGNORE_INTERNAL = 0x4,
    DXGI_DEBUG_RLO_ALL = 0x7
} DXGI_DEBUG_RLO_FLAGS;
typedef enum DXGI_INFO_QUEUE_MESSAGE_CATEGORY {
    DXGI_INFO_QUEUE_MESSAGE_CATEGORY_UNKNOWN = 0,
    DXGI_INFO_QUEUE_MESSAGE_CATEGORY_MISCELLANEOUS = 1,
    DXGI_INFO_QUEUE_MESSAGE_CATEGORY_INITIALIZATION = 2,
    DXGI_INFO_QUEUE_MESSAGE_CATEGORY_CLEANUP = 3,
    DXGI_INFO_QUEUE_MESSAGE_CATEGORY_COMPILATION = 4,
    DXGI_INFO_QUEUE_MESSAGE_CATEGORY_STATE_CREATION = 5,
    DXGI_INFO_QUEUE_MESSAGE_CATEGORY_STATE_SETTING = 6,
    DXGI_INFO_QUEUE_MESSAGE_CATEGORY_STATE_GETTING = 7,
    DXGI_INFO_QUEUE_MESSAGE_CATEGORY_RESOURCE_MANIPULATION = 8,
    DXGI_INFO_QUEUE_MESSAGE_CATEGORY_EXECUTION = 9,
    DXGI_INFO_QUEUE_MESSAGE_CATEGORY_SHADER = 10
} DXGI_INFO_QUEUE_MESSAGE_CATEGORY;
typedef enum DXGI_INFO_QUEUE_MESSAGE_SEVERITY {
    DXGI_INFO_QUEUE_MESSAGE_SEVERITY_CORRUPTION = 0,
    DXGI_INFO_QUEUE_MESSAGE_SEVERITY_ERROR = 1,
    DXGI_INFO_QUEUE_MESSAGE_SEVERITY_WARNING = 2,
    DXGI_INFO_QUEUE_MESSAGE_SEVERITY_INFO = 3,
    DXGI_INFO_QUEUE_MESSAGE_SEVERITY_MESSAGE = 4
} DXGI_INFO_QUEUE_MESSAGE_SEVERITY;
typedef int DXGI_INFO_QUEUE_MESSAGE_ID;
#define DXGI_INFO_QUEUE_MESSAGE_ID_STRING_FROM_APPLICATION 0
typedef struct DXGI_INFO_QUEUE_MESSAGE {
    DXGI_DEBUG_ID Producer;
    DXGI_INFO_QUEUE_MESSAGE_CATEGORY Category;
    DXGI_INFO_QUEUE_MESSAGE_SEVERITY Severity;
    DXGI_INFO_QUEUE_MESSAGE_ID ID;
    const char *pDescription;
    SIZE_T DescriptionByteLength;
} DXGI_INFO_QUEUE_MESSAGE;
typedef struct DXGI_INFO_QUEUE_FILTER_DESC {
    UINT NumCategories;
    DXGI_INFO_QUEUE_MESSAGE_CATEGORY *pCategoryList;
    UINT NumSeverities;
    DXGI_INFO_QUEUE_MESSAGE_SEVERITY *pSeverityList;
    UINT NumIDs;
    DXGI_INFO_QUEUE_MESSAGE_ID *pIDList;
} DXGI_INFO_QUEUE_FILTER_DESC;
typedef struct DXGI_INFO_QUEUE_FILTER {
    DXGI_INFO_QUEUE_FILTER_DESC AllowList;
    DXGI_INFO_QUEUE_FILTER_DESC DenyList;
} DXGI_INFO_QUEUE_FILTER;
#define DXGI_INFO_QUEUE_DEFAULT_MESSAGE_COUNT_LIMIT 1024
HRESULT WINAPI DXGIGetDebugInterface(REFIID riid, void **ppDebug);
/*****************************************************************************
 * IDXGIInfoQueue interface
 */
#ifndef __IDXGIInfoQueue_INTERFACE_DEFINED__
#define __IDXGIInfoQueue_INTERFACE_DEFINED__

DEFINE_GUID(IID_IDXGIInfoQueue, 0xd67441c7, 0x672a, 0x476f, 0x9e,0x82, 0xcd,0x55,0xb4,0x49,0x49,0xce);
#if defined(__cplusplus) && !defined(CINTERFACE)
MIDL_INTERFACE("d67441c7-672a-476f-9e82-cd55b44949ce")
IDXGIInfoQueue : public IUnknown
{
    virtual HRESULT STDMETHODCALLTYPE SetMessageCountLimit(
        DXGI_DEBUG_ID producer,
        UINT64 limit) = 0;

    virtual void STDMETHODCALLTYPE ClearStoredMessages(
        DXGI_DEBUG_ID producer) = 0;

    virtual HRESULT STDMETHODCALLTYPE GetMessage(
        DXGI_DEBUG_ID producer,
        UINT64 index,
        DXGI_INFO_QUEUE_MESSAGE *message,
        SIZE_T *length) = 0;

    virtual UINT64 STDMETHODCALLTYPE GetNumStoredMessagesAllowedByRetrievalFilters(
        DXGI_DEBUG_ID producer) = 0;

    virtual UINT64 STDMETHODCALLTYPE GetNumStoredMessages(
        DXGI_DEBUG_ID producer) = 0;

    virtual UINT64 STDMETHODCALLTYPE GetNumMessagesDiscardedByMessageCountLimit(
        DXGI_DEBUG_ID producer) = 0;

    virtual UINT64 STDMETHODCALLTYPE GetMessageCountLimit(
        DXGI_DEBUG_ID producer) = 0;

    virtual UINT64 STDMETHODCALLTYPE GetNumMessagesAllowedByStorageFilter(
        DXGI_DEBUG_ID producer) = 0;

    virtual UINT64 STDMETHODCALLTYPE GetNumMessagesDeniedByStorageFilter(
        DXGI_DEBUG_ID producer) = 0;

    virtual HRESULT STDMETHODCALLTYPE AddStorageFilterEntries(
        DXGI_DEBUG_ID producer,
        DXGI_INFO_QUEUE_FILTER *filter) = 0;

    virtual HRESULT STDMETHODCALLTYPE GetStorageFilter(
        DXGI_DEBUG_ID producer,
        DXGI_INFO_QUEUE_FILTER *filter,
        SIZE_T *length) = 0;

    virtual void STDMETHODCALLTYPE ClearStorageFilter(
        DXGI_DEBUG_ID producer) = 0;

    virtual HRESULT STDMETHODCALLTYPE PushEmptyStorageFilter(
        DXGI_DEBUG_ID producer) = 0;

    virtual HRESULT STDMETHODCALLTYPE PushDenyAllStorageFilter(
        DXGI_DEBUG_ID producer) = 0;

    virtual HRESULT STDMETHODCALLTYPE PushCopyOfStorageFilter(
        DXGI_DEBUG_ID producer) = 0;

    virtual HRESULT STDMETHODCALLTYPE PushStorageFilter(
        DXGI_DEBUG_ID producer,
        DXGI_INFO_QUEUE_FILTER *filter) = 0;

    virtual void STDMETHODCALLTYPE PopStorageFilter(
        DXGI_DEBUG_ID producer) = 0;

    virtual UINT STDMETHODCALLTYPE GetStorageFilterStackSize(
        DXGI_DEBUG_ID producer) = 0;

    virtual HRESULT STDMETHODCALLTYPE AddRetrievalFilterEntries(
        DXGI_DEBUG_ID producer,
        DXGI_INFO_QUEUE_FILTER *filter) = 0;

    virtual HRESULT STDMETHODCALLTYPE GetRetrievalFilter(
        DXGI_DEBUG_ID producer,
        DXGI_INFO_QUEUE_FILTER *filter,
        SIZE_T *length) = 0;

    virtual void STDMETHODCALLTYPE ClearRetrievalFilter(
        DXGI_DEBUG_ID producer) = 0;

    virtual HRESULT STDMETHODCALLTYPE PushEmptyRetrievalFilter(
        DXGI_DEBUG_ID producer) = 0;

    virtual HRESULT STDMETHODCALLTYPE PushDenyAllRetrievalFilter(
        DXGI_DEBUG_ID producer) = 0;

    virtual HRESULT STDMETHODCALLTYPE PushCopyOfRetrievalFilter(
        DXGI_DEBUG_ID producer) = 0;

    virtual HRESULT STDMETHODCALLTYPE PushRetrievalFilter(
        DXGI_DEBUG_ID producer,
        DXGI_INFO_QUEUE_FILTER *filter) = 0;

    virtual void STDMETHODCALLTYPE PopRetrievalFilter(
        DXGI_DEBUG_ID producer) = 0;

    virtual UINT STDMETHODCALLTYPE GetRetrievalFilterStackSize(
        DXGI_DEBUG_ID producer) = 0;

    virtual HRESULT STDMETHODCALLTYPE AddMessage(
        DXGI_DEBUG_ID producer,
        DXGI_INFO_QUEUE_MESSAGE_CATEGORY category,
        DXGI_INFO_QUEUE_MESSAGE_SEVERITY severity,
        DXGI_INFO_QUEUE_MESSAGE_ID id,
        LPCSTR description) = 0;

    virtual HRESULT STDMETHODCALLTYPE AddApplicationMessage(
        DXGI_INFO_QUEUE_MESSAGE_SEVERITY severity,
        LPCSTR description) = 0;

    virtual HRESULT STDMETHODCALLTYPE SetBreakOnCategory(
        DXGI_DEBUG_ID producer,
        DXGI_INFO_QUEUE_MESSAGE_CATEGORY category,
        BOOL enable) = 0;

    virtual HRESULT STDMETHODCALLTYPE SetBreakOnSeverity(
        DXGI_DEBUG_ID producer,
        DXGI_INFO_QUEUE_MESSAGE_SEVERITY severity,
        BOOL enable) = 0;

    virtual HRESULT STDMETHODCALLTYPE SetBreakOnID(
        DXGI_DEBUG_ID producer,
        DXGI_INFO_QUEUE_MESSAGE_ID id,
        BOOL enable) = 0;

    virtual BOOL STDMETHODCALLTYPE GetBreakOnCategory(
        DXGI_DEBUG_ID producer,
        DXGI_INFO_QUEUE_MESSAGE_CATEGORY category) = 0;

    virtual BOOL STDMETHODCALLTYPE GetBreakOnSeverity(
        DXGI_DEBUG_ID producer,
        DXGI_INFO_QUEUE_MESSAGE_SEVERITY severity) = 0;

    virtual BOOL STDMETHODCALLTYPE GetBreakOnID(
        DXGI_DEBUG_ID producer,
        DXGI_INFO_QUEUE_MESSAGE_ID id) = 0;

    virtual void STDMETHODCALLTYPE SetMuteDebugOutput(
        DXGI_DEBUG_ID producer,
        BOOL mute) = 0;

    virtual BOOL STDMETHODCALLTYPE GetMuteDebugOutput(
        DXGI_DEBUG_ID producer) = 0;

};
#ifdef __CRT_UUID_DECL
__CRT_UUID_DECL(IDXGIInfoQueue, 0xd67441c7, 0x672a, 0x476f, 0x9e,0x82, 0xcd,0x55,0xb4,0x49,0x49,0xce)
#endif
#else
typedef struct IDXGIInfoQueueVtbl {
    BEGIN_INTERFACE

    /*** IUnknown methods ***/
    HRESULT (STDMETHODCALLTYPE *QueryInterface)(
        IDXGIInfoQueue *This,
        REFIID riid,
        void **ppvObject);

    ULONG (STDMETHODCALLTYPE *AddRef)(
        IDXGIInfoQueue *This);

    ULONG (STDMETHODCALLTYPE *Release)(
        IDXGIInfoQueue *This);

    /*** IDXGIInfoQueue methods ***/
    HRESULT (STDMETHODCALLTYPE *SetMessageCountLimit)(
        IDXGIInfoQueue *This,
        DXGI_DEBUG_ID producer,
        UINT64 limit);

    void (STDMETHODCALLTYPE *ClearStoredMessages)(
        IDXGIInfoQueue *This,
        DXGI_DEBUG_ID producer);

    HRESULT (STDMETHODCALLTYPE *GetMessage)(
        IDXGIInfoQueue *This,
        DXGI_DEBUG_ID producer,
        UINT64 index,
        DXGI_INFO_QUEUE_MESSAGE *message,
        SIZE_T *length);

    UINT64 (STDMETHODCALLTYPE *GetNumStoredMessagesAllowedByRetrievalFilters)(
        IDXGIInfoQueue *This,
        DXGI_DEBUG_ID producer);

    UINT64 (STDMETHODCALLTYPE *GetNumStoredMessages)(
        IDXGIInfoQueue *This,
        DXGI_DEBUG_ID producer);

    UINT64 (STDMETHODCALLTYPE *GetNumMessagesDiscardedByMessageCountLimit)(
        IDXGIInfoQueue *This,
        DXGI_DEBUG_ID producer);

    UINT64 (STDMETHODCALLTYPE *GetMessageCountLimit)(
        IDXGIInfoQueue *This,
        DXGI_DEBUG_ID producer);

    UINT64 (STDMETHODCALLTYPE *GetNumMessagesAllowedByStorageFilter)(
        IDXGIInfoQueue *This,
        DXGI_DEBUG_ID producer);

    UINT64 (STDMETHODCALLTYPE *GetNumMessagesDeniedByStorageFilter)(
        IDXGIInfoQueue *This,
        DXGI_DEBUG_ID producer);

    HRESULT (STDMETHODCALLTYPE *AddStorageFilterEntries)(
        IDXGIInfoQueue *This,
        DXGI_DEBUG_ID producer,
        DXGI_INFO_QUEUE_FILTER *filter);

    HRESULT (STDMETHODCALLTYPE *GetStorageFilter)(
        IDXGIInfoQueue *This,
        DXGI_DEBUG_ID producer,
        DXGI_INFO_QUEUE_FILTER *filter,
        SIZE_T *length);

    void (STDMETHODCALLTYPE *ClearStorageFilter)(
        IDXGIInfoQueue *This,
        DXGI_DEBUG_ID producer);

    HRESULT (STDMETHODCALLTYPE *PushEmptyStorageFilter)(
        IDXGIInfoQueue *This,
        DXGI_DEBUG_ID producer);

    HRESULT (STDMETHODCALLTYPE *PushDenyAllStorageFilter)(
        IDXGIInfoQueue *This,
        DXGI_DEBUG_ID producer);

    HRESULT (STDMETHODCALLTYPE *PushCopyOfStorageFilter)(
        IDXGIInfoQueue *This,
        DXGI_DEBUG_ID producer);

    HRESULT (STDMETHODCALLTYPE *PushStorageFilter)(
        IDXGIInfoQueue *This,
        DXGI_DEBUG_ID producer,
        DXGI_INFO_QUEUE_FILTER *filter);

    void (STDMETHODCALLTYPE *PopStorageFilter)(
        IDXGIInfoQueue *This,
        DXGI_DEBUG_ID producer);

    UINT (STDMETHODCALLTYPE *GetStorageFilterStackSize)(
        IDXGIInfoQueue *This,
        DXGI_DEBUG_ID producer);

    HRESULT (STDMETHODCALLTYPE *AddRetrievalFilterEntries)(
        IDXGIInfoQueue *This,
        DXGI_DEBUG_ID producer,
        DXGI_INFO_QUEUE_FILTER *filter);

    HRESULT (STDMETHODCALLTYPE *GetRetrievalFilter)(
        IDXGIInfoQueue *This,
        DXGI_DEBUG_ID producer,
        DXGI_INFO_QUEUE_FILTER *filter,
        SIZE_T *length);

    void (STDMETHODCALLTYPE *ClearRetrievalFilter)(
        IDXGIInfoQueue *This,
        DXGI_DEBUG_ID producer);

    HRESULT (STDMETHODCALLTYPE *PushEmptyRetrievalFilter)(
        IDXGIInfoQueue *This,
        DXGI_DEBUG_ID producer);

    HRESULT (STDMETHODCALLTYPE *PushDenyAllRetrievalFilter)(
        IDXGIInfoQueue *This,
        DXGI_DEBUG_ID producer);

    HRESULT (STDMETHODCALLTYPE *PushCopyOfRetrievalFilter)(
        IDXGIInfoQueue *This,
        DXGI_DEBUG_ID producer);

    HRESULT (STDMETHODCALLTYPE *PushRetrievalFilter)(
        IDXGIInfoQueue *This,
        DXGI_DEBUG_ID producer,
        DXGI_INFO_QUEUE_FILTER *filter);

    void (STDMETHODCALLTYPE *PopRetrievalFilter)(
        IDXGIInfoQueue *This,
        DXGI_DEBUG_ID producer);

    UINT (STDMETHODCALLTYPE *GetRetrievalFilterStackSize)(
        IDXGIInfoQueue *This,
        DXGI_DEBUG_ID producer);

    HRESULT (STDMETHODCALLTYPE *AddMessage)(
        IDXGIInfoQueue *This,
        DXGI_DEBUG_ID producer,
        DXGI_INFO_QUEUE_MESSAGE_CATEGORY category,
        DXGI_INFO_QUEUE_MESSAGE_SEVERITY severity,
        DXGI_INFO_QUEUE_MESSAGE_ID id,
        LPCSTR description);

    HRESULT (STDMETHODCALLTYPE *AddApplicationMessage)(
        IDXGIInfoQueue *This,
        DXGI_INFO_QUEUE_MESSAGE_SEVERITY severity,
        LPCSTR description);

    HRESULT (STDMETHODCALLTYPE *SetBreakOnCategory)(
        IDXGIInfoQueue *This,
        DXGI_DEBUG_ID producer,
        DXGI_INFO_QUEUE_MESSAGE_CATEGORY category,
        BOOL enable);

    HRESULT (STDMETHODCALLTYPE *SetBreakOnSeverity)(
        IDXGIInfoQueue *This,
        DXGI_DEBUG_ID producer,
        DXGI_INFO_QUEUE_MESSAGE_SEVERITY severity,
        BOOL enable);

    HRESULT (STDMETHODCALLTYPE *SetBreakOnID)(
        IDXGIInfoQueue *This,
        DXGI_DEBUG_ID producer,
        DXGI_INFO_QUEUE_MESSAGE_ID id,
        BOOL enable);

    BOOL (STDMETHODCALLTYPE *GetBreakOnCategory)(
        IDXGIInfoQueue *This,
        DXGI_DEBUG_ID producer,
        DXGI_INFO_QUEUE_MESSAGE_CATEGORY category);

    BOOL (STDMETHODCALLTYPE *GetBreakOnSeverity)(
        IDXGIInfoQueue *This,
        DXGI_DEBUG_ID producer,
        DXGI_INFO_QUEUE_MESSAGE_SEVERITY severity);

    BOOL (STDMETHODCALLTYPE *GetBreakOnID)(
        IDXGIInfoQueue *This,
        DXGI_DEBUG_ID producer,
        DXGI_INFO_QUEUE_MESSAGE_ID id);

    void (STDMETHODCALLTYPE *SetMuteDebugOutput)(
        IDXGIInfoQueue *This,
        DXGI_DEBUG_ID producer,
        BOOL mute);

    BOOL (STDMETHODCALLTYPE *GetMuteDebugOutput)(
        IDXGIInfoQueue *This,
        DXGI_DEBUG_ID producer);

    END_INTERFACE
} IDXGIInfoQueueVtbl;

interface IDXGIInfoQueue {
    CONST_VTBL IDXGIInfoQueueVtbl* lpVtbl;
};

#ifdef COBJMACROS
#ifndef WIDL_C_INLINE_WRAPPERS
/*** IUnknown methods ***/
#define IDXGIInfoQueue_QueryInterface(This,riid,ppvObject) (This)->lpVtbl->QueryInterface(This,riid,ppvObject)
#define IDXGIInfoQueue_AddRef(This) (This)->lpVtbl->AddRef(This)
#define IDXGIInfoQueue_Release(This) (This)->lpVtbl->Release(This)
/*** IDXGIInfoQueue methods ***/
#define IDXGIInfoQueue_SetMessageCountLimit(This,producer,limit) (This)->lpVtbl->SetMessageCountLimit(This,producer,limit)
#define IDXGIInfoQueue_ClearStoredMessages(This,producer) (This)->lpVtbl->ClearStoredMessages(This,producer)
#define IDXGIInfoQueue_GetMessage(This,producer,index,message,length) (This)->lpVtbl->GetMessage(This,producer,index,message,length)
#define IDXGIInfoQueue_GetNumStoredMessagesAllowedByRetrievalFilters(This,producer) (This)->lpVtbl->GetNumStoredMessagesAllowedByRetrievalFilters(This,producer)
#define IDXGIInfoQueue_GetNumStoredMessages(This,producer) (This)->lpVtbl->GetNumStoredMessages(This,producer)
#define IDXGIInfoQueue_GetNumMessagesDiscardedByMessageCountLimit(This,producer) (This)->lpVtbl->GetNumMessagesDiscardedByMessageCountLimit(This,producer)
#define IDXGIInfoQueue_GetMessageCountLimit(This,producer) (This)->lpVtbl->GetMessageCountLimit(This,producer)
#define IDXGIInfoQueue_GetNumMessagesAllowedByStorageFilter(This,producer) (This)->lpVtbl->GetNumMessagesAllowedByStorageFilter(This,producer)
#define IDXGIInfoQueue_GetNumMessagesDeniedByStorageFilter(This,producer) (This)->lpVtbl->GetNumMessagesDeniedByStorageFilter(This,producer)
#define IDXGIInfoQueue_AddStorageFilterEntries(This,producer,filter) (This)->lpVtbl->AddStorageFilterEntries(This,producer,filter)
#define IDXGIInfoQueue_GetStorageFilter(This,producer,filter,length) (This)->lpVtbl->GetStorageFilter(This,producer,filter,length)
#define IDXGIInfoQueue_ClearStorageFilter(This,producer) (This)->lpVtbl->ClearStorageFilter(This,producer)
#define IDXGIInfoQueue_PushEmptyStorageFilter(This,producer) (This)->lpVtbl->PushEmptyStorageFilter(This,producer)
#define IDXGIInfoQueue_PushDenyAllStorageFilter(This,producer) (This)->lpVtbl->PushDenyAllStorageFilter(This,producer)
#define IDXGIInfoQueue_PushCopyOfStorageFilter(This,producer) (This)->lpVtbl->PushCopyOfStorageFilter(This,producer)
#define IDXGIInfoQueue_PushStorageFilter(This,producer,filter) (This)->lpVtbl->PushStorageFilter(This,producer,filter)
#define IDXGIInfoQueue_PopStorageFilter(This,producer) (This)->lpVtbl->PopStorageFilter(This,producer)
#define IDXGIInfoQueue_GetStorageFilterStackSize(This,producer) (This)->lpVtbl->GetStorageFilterStackSize(This,producer)
#define IDXGIInfoQueue_AddRetrievalFilterEntries(This,producer,filter) (This)->lpVtbl->AddRetrievalFilterEntries(This,producer,filter)
#define IDXGIInfoQueue_GetRetrievalFilter(This,producer,filter,length) (This)->lpVtbl->GetRetrievalFilter(This,producer,filter,length)
#define IDXGIInfoQueue_ClearRetrievalFilter(This,producer) (This)->lpVtbl->ClearRetrievalFilter(This,producer)
#define IDXGIInfoQueue_PushEmptyRetrievalFilter(This,producer) (This)->lpVtbl->PushEmptyRetrievalFilter(This,producer)
#define IDXGIInfoQueue_PushDenyAllRetrievalFilter(This,producer) (This)->lpVtbl->PushDenyAllRetrievalFilter(This,producer)
#define IDXGIInfoQueue_PushCopyOfRetrievalFilter(This,producer) (This)->lpVtbl->PushCopyOfRetrievalFilter(This,producer)
#define IDXGIInfoQueue_PushRetrievalFilter(This,producer,filter) (This)->lpVtbl->PushRetrievalFilter(This,producer,filter)
#define IDXGIInfoQueue_PopRetrievalFilter(This,producer) (This)->lpVtbl->PopRetrievalFilter(This,producer)
#define IDXGIInfoQueue_GetRetrievalFilterStackSize(This,producer) (This)->lpVtbl->GetRetrievalFilterStackSize(This,producer)
#define IDXGIInfoQueue_AddMessage(This,producer,category,severity,id,description) (This)->lpVtbl->AddMessage(This,producer,category,severity,id,description)
#define IDXGIInfoQueue_AddApplicationMessage(This,severity,description) (This)->lpVtbl->AddApplicationMessage(This,severity,description)
#define IDXGIInfoQueue_SetBreakOnCategory(This,producer,category,enable) (This)->lpVtbl->SetBreakOnCategory(This,producer,category,enable)
#define IDXGIInfoQueue_SetBreakOnSeverity(This,producer,severity,enable) (This)->lpVtbl->SetBreakOnSeverity(This,producer,severity,enable)
#define IDXGIInfoQueue_SetBreakOnID(This,producer,id,enable) (This)->lpVtbl->SetBreakOnID(This,producer,id,enable)
#define IDXGIInfoQueue_GetBreakOnCategory(This,producer,category) (This)->lpVtbl->GetBreakOnCategory(This,producer,category)
#define IDXGIInfoQueue_GetBreakOnSeverity(This,producer,severity) (This)->lpVtbl->GetBreakOnSeverity(This,producer,severity)
#define IDXGIInfoQueue_GetBreakOnID(This,producer,id) (This)->lpVtbl->GetBreakOnID(This,producer,id)
#define IDXGIInfoQueue_SetMuteDebugOutput(This,producer,mute) (This)->lpVtbl->SetMuteDebugOutput(This,producer,mute)
#define IDXGIInfoQueue_GetMuteDebugOutput(This,producer) (This)->lpVtbl->GetMuteDebugOutput(This,producer)
#else
/*** IUnknown methods ***/
static __WIDL_INLINE HRESULT IDXGIInfoQueue_QueryInterface(IDXGIInfoQueue* This,REFIID riid,void **ppvObject) {
    return This->lpVtbl->QueryInterface(This,riid,ppvObject);
}
static __WIDL_INLINE ULONG IDXGIInfoQueue_AddRef(IDXGIInfoQueue* This) {
    return This->lpVtbl->AddRef(This);
}
static __WIDL_INLINE ULONG IDXGIInfoQueue_Release(IDXGIInfoQueue* This) {
    return This->lpVtbl->Release(This);
}
/*** IDXGIInfoQueue methods ***/
static __WIDL_INLINE HRESULT IDXGIInfoQueue_SetMessageCountLimit(IDXGIInfoQueue* This,DXGI_DEBUG_ID producer,UINT64 limit) {
    return This->lpVtbl->SetMessageCountLimit(This,producer,limit);
}
static __WIDL_INLINE void IDXGIInfoQueue_ClearStoredMessages(IDXGIInfoQueue* This,DXGI_DEBUG_ID producer) {
    This->lpVtbl->ClearStoredMessages(This,producer);
}
static __WIDL_INLINE HRESULT IDXGIInfoQueue_GetMessage(IDXGIInfoQueue* This,DXGI_DEBUG_ID producer,UINT64 index,DXGI_INFO_QUEUE_MESSAGE *message,SIZE_T *length) {
    return This->lpVtbl->GetMessage(This,producer,index,message,length);
}
static __WIDL_INLINE UINT64 IDXGIInfoQueue_GetNumStoredMessagesAllowedByRetrievalFilters(IDXGIInfoQueue* This,DXGI_DEBUG_ID producer) {
    return This->lpVtbl->GetNumStoredMessagesAllowedByRetrievalFilters(This,producer);
}
static __WIDL_INLINE UINT64 IDXGIInfoQueue_GetNumStoredMessages(IDXGIInfoQueue* This,DXGI_DEBUG_ID producer) {
    return This->lpVtbl->GetNumStoredMessages(This,producer);
}
static __WIDL_INLINE UINT64 IDXGIInfoQueue_GetNumMessagesDiscardedByMessageCountLimit(IDXGIInfoQueue* This,DXGI_DEBUG_ID producer) {
    return This->lpVtbl->GetNumMessagesDiscardedByMessageCountLimit(This,producer);
}
static __WIDL_INLINE UINT64 IDXGIInfoQueue_GetMessageCountLimit(IDXGIInfoQueue* This,DXGI_DEBUG_ID producer) {
    return This->lpVtbl->GetMessageCountLimit(This,producer);
}
static __WIDL_INLINE UINT64 IDXGIInfoQueue_GetNumMessagesAllowedByStorageFilter(IDXGIInfoQueue* This,DXGI_DEBUG_ID producer) {
    return This->lpVtbl->GetNumMessagesAllowedByStorageFilter(This,producer);
}
static __WIDL_INLINE UINT64 IDXGIInfoQueue_GetNumMessagesDeniedByStorageFilter(IDXGIInfoQueue* This,DXGI_DEBUG_ID producer) {
    return This->lpVtbl->GetNumMessagesDeniedByStorageFilter(This,producer);
}
static __WIDL_INLINE HRESULT IDXGIInfoQueue_AddStorageFilterEntries(IDXGIInfoQueue* This,DXGI_DEBUG_ID producer,DXGI_INFO_QUEUE_FILTER *filter) {
    return This->lpVtbl->AddStorageFilterEntries(This,producer,filter);
}
static __WIDL_INLINE HRESULT IDXGIInfoQueue_GetStorageFilter(IDXGIInfoQueue* This,DXGI_DEBUG_ID producer,DXGI_INFO_QUEUE_FILTER *filter,SIZE_T *length) {
    return This->lpVtbl->GetStorageFilter(This,producer,filter,length);
}
static __WIDL_INLINE void IDXGIInfoQueue_ClearStorageFilter(IDXGIInfoQueue* This,DXGI_DEBUG_ID producer) {
    This->lpVtbl->ClearStorageFilter(This,producer);
}
static __WIDL_INLINE HRESULT IDXGIInfoQueue_PushEmptyStorageFilter(IDXGIInfoQueue* This,DXGI_DEBUG_ID producer) {
    return This->lpVtbl->PushEmptyStorageFilter(This,producer);
}
static __WIDL_INLINE HRESULT IDXGIInfoQueue_PushDenyAllStorageFilter(IDXGIInfoQueue* This,DXGI_DEBUG_ID producer) {
    return This->lpVtbl->PushDenyAllStorageFilter(This,producer);
}
static __WIDL_INLINE HRESULT IDXGIInfoQueue_PushCopyOfStorageFilter(IDXGIInfoQueue* This,DXGI_DEBUG_ID producer) {
    return This->lpVtbl->PushCopyOfStorageFilter(This,producer);
}
static __WIDL_INLINE HRESULT IDXGIInfoQueue_PushStorageFilter(IDXGIInfoQueue* This,DXGI_DEBUG_ID producer,DXGI_INFO_QUEUE_FILTER *filter) {
    return This->lpVtbl->PushStorageFilter(This,producer,filter);
}
static __WIDL_INLINE void IDXGIInfoQueue_PopStorageFilter(IDXGIInfoQueue* This,DXGI_DEBUG_ID producer) {
    This->lpVtbl->PopStorageFilter(This,producer);
}
static __WIDL_INLINE UINT IDXGIInfoQueue_GetStorageFilterStackSize(IDXGIInfoQueue* This,DXGI_DEBUG_ID producer) {
    return This->lpVtbl->GetStorageFilterStackSize(This,producer);
}
static __WIDL_INLINE HRESULT IDXGIInfoQueue_AddRetrievalFilterEntries(IDXGIInfoQueue* This,DXGI_DEBUG_ID producer,DXGI_INFO_QUEUE_FILTER *filter) {
    return This->lpVtbl->AddRetrievalFilterEntries(This,producer,filter);
}
static __WIDL_INLINE HRESULT IDXGIInfoQueue_GetRetrievalFilter(IDXGIInfoQueue* This,DXGI_DEBUG_ID producer,DXGI_INFO_QUEUE_FILTER *filter,SIZE_T *length) {
    return This->lpVtbl->GetRetrievalFilter(This,producer,filter,length);
}
static __WIDL_INLINE void IDXGIInfoQueue_ClearRetrievalFilter(IDXGIInfoQueue* This,DXGI_DEBUG_ID producer) {
    This->lpVtbl->ClearRetrievalFilter(This,producer);
}
static __WIDL_INLINE HRESULT IDXGIInfoQueue_PushEmptyRetrievalFilter(IDXGIInfoQueue* This,DXGI_DEBUG_ID producer) {
    return This->lpVtbl->PushEmptyRetrievalFilter(This,producer);
}
static __WIDL_INLINE HRESULT IDXGIInfoQueue_PushDenyAllRetrievalFilter(IDXGIInfoQueue* This,DXGI_DEBUG_ID producer) {
    return This->lpVtbl->PushDenyAllRetrievalFilter(This,producer);
}
static __WIDL_INLINE HRESULT IDXGIInfoQueue_PushCopyOfRetrievalFilter(IDXGIInfoQueue* This,DXGI_DEBUG_ID producer) {
    return This->lpVtbl->PushCopyOfRetrievalFilter(This,producer);
}
static __WIDL_INLINE HRESULT IDXGIInfoQueue_PushRetrievalFilter(IDXGIInfoQueue* This,DXGI_DEBUG_ID producer,DXGI_INFO_QUEUE_FILTER *filter) {
    return This->lpVtbl->PushRetrievalFilter(This,producer,filter);
}
static __WIDL_INLINE void IDXGIInfoQueue_PopRetrievalFilter(IDXGIInfoQueue* This,DXGI_DEBUG_ID producer) {
    This->lpVtbl->PopRetrievalFilter(This,producer);
}
static __WIDL_INLINE UINT IDXGIInfoQueue_GetRetrievalFilterStackSize(IDXGIInfoQueue* This,DXGI_DEBUG_ID producer) {
    return This->lpVtbl->GetRetrievalFilterStackSize(This,producer);
}
static __WIDL_INLINE HRESULT IDXGIInfoQueue_AddMessage(IDXGIInfoQueue* This,DXGI_DEBUG_ID producer,DXGI_INFO_QUEUE_MESSAGE_CATEGORY category,DXGI_INFO_QUEUE_MESSAGE_SEVERITY severity,DXGI_INFO_QUEUE_MESSAGE_ID id,LPCSTR description) {
    return This->lpVtbl->AddMessage(This,producer,category,severity,id,description);
}
static __WIDL_INLINE HRESULT IDXGIInfoQueue_AddApplicationMessage(IDXGIInfoQueue* This,DXGI_INFO_QUEUE_MESSAGE_SEVERITY severity,LPCSTR description) {
    return This->lpVtbl->AddApplicationMessage(This,severity,description);
}
static __WIDL_INLINE HRESULT IDXGIInfoQueue_SetBreakOnCategory(IDXGIInfoQueue* This,DXGI_DEBUG_ID producer,DXGI_INFO_QUEUE_MESSAGE_CATEGORY category,BOOL enable) {
    return This->lpVtbl->SetBreakOnCategory(This,producer,category,enable);
}
static __WIDL_INLINE HRESULT IDXGIInfoQueue_SetBreakOnSeverity(IDXGIInfoQueue* This,DXGI_DEBUG_ID producer,DXGI_INFO_QUEUE_MESSAGE_SEVERITY severity,BOOL enable) {
    return This->lpVtbl->SetBreakOnSeverity(This,producer,severity,enable);
}
static __WIDL_INLINE HRESULT IDXGIInfoQueue_SetBreakOnID(IDXGIInfoQueue* This,DXGI_DEBUG_ID producer,DXGI_INFO_QUEUE_MESSAGE_ID id,BOOL enable) {
    return This->lpVtbl->SetBreakOnID(This,producer,id,enable);
}
static __WIDL_INLINE BOOL IDXGIInfoQueue_GetBreakOnCategory(IDXGIInfoQueue* This,DXGI_DEBUG_ID producer,DXGI_INFO_QUEUE_MESSAGE_CATEGORY category) {
    return This->lpVtbl->GetBreakOnCategory(This,producer,category);
}
static __WIDL_INLINE BOOL IDXGIInfoQueue_GetBreakOnSeverity(IDXGIInfoQueue* This,DXGI_DEBUG_ID producer,DXGI_INFO_QUEUE_MESSAGE_SEVERITY severity) {
    return This->lpVtbl->GetBreakOnSeverity(This,producer,severity);
}
static __WIDL_INLINE BOOL IDXGIInfoQueue_GetBreakOnID(IDXGIInfoQueue* This,DXGI_DEBUG_ID producer,DXGI_INFO_QUEUE_MESSAGE_ID id) {
    return This->lpVtbl->GetBreakOnID(This,producer,id);
}
static __WIDL_INLINE void IDXGIInfoQueue_SetMuteDebugOutput(IDXGIInfoQueue* This,DXGI_DEBUG_ID producer,BOOL mute) {
    This->lpVtbl->SetMuteDebugOutput(This,producer,mute);
}
static __WIDL_INLINE BOOL IDXGIInfoQueue_GetMuteDebugOutput(IDXGIInfoQueue* This,DXGI_DEBUG_ID producer) {
    return This->lpVtbl->GetMuteDebugOutput(This,producer);
}
#endif
#endif

#endif


#endif  /* __IDXGIInfoQueue_INTERFACE_DEFINED__ */

/*****************************************************************************
 * IDXGIDebug interface
 */
#ifndef __IDXGIDebug_INTERFACE_DEFINED__
#define __IDXGIDebug_INTERFACE_DEFINED__

DEFINE_GUID(IID_IDXGIDebug, 0x119e7452, 0xde9e, 0x40fe, 0x88,0x06, 0x88,0xf9,0x0c,0x12,0xb4,0x41);
#if defined(__cplusplus) && !defined(CINTERFACE)
MIDL_INTERFACE("119e7452-de9e-40fe-8806-88f90c12b441")
IDXGIDebug : public IUnknown
{
    virtual HRESULT STDMETHODCALLTYPE ReportLiveObjects(
        GUID apiid,
        DXGI_DEBUG_RLO_FLAGS flags) = 0;

};
#ifdef __CRT_UUID_DECL
__CRT_UUID_DECL(IDXGIDebug, 0x119e7452, 0xde9e, 0x40fe, 0x88,0x06, 0x88,0xf9,0x0c,0x12,0xb4,0x41)
#endif
#else
typedef struct IDXGIDebugVtbl {
    BEGIN_INTERFACE

    /*** IUnknown methods ***/
    HRESULT (STDMETHODCALLTYPE *QueryInterface)(
        IDXGIDebug *This,
        REFIID riid,
        void **ppvObject);

    ULONG (STDMETHODCALLTYPE *AddRef)(
        IDXGIDebug *This);

    ULONG (STDMETHODCALLTYPE *Release)(
        IDXGIDebug *This);

    /*** IDXGIDebug methods ***/
    HRESULT (STDMETHODCALLTYPE *ReportLiveObjects)(
        IDXGIDebug *This,
        GUID apiid,
        DXGI_DEBUG_RLO_FLAGS flags);

    END_INTERFACE
} IDXGIDebugVtbl;

interface IDXGIDebug {
    CONST_VTBL IDXGIDebugVtbl* lpVtbl;
};

#ifdef COBJMACROS
#ifndef WIDL_C_INLINE_WRAPPERS
/*** IUnknown methods ***/
#define IDXGIDebug_QueryInterface(This,riid,ppvObject) (This)->lpVtbl->QueryInterface(This,riid,ppvObject)
#define IDXGIDebug_AddRef(This) (This)->lpVtbl->AddRef(This)
#define IDXGIDebug_Release(This) (This)->lpVtbl->Release(This)
/*** IDXGIDebug methods ***/
#define IDXGIDebug_ReportLiveObjects(This,apiid,flags) (This)->lpVtbl->ReportLiveObjects(This,apiid,flags)
#else
/*** IUnknown methods ***/
static __WIDL_INLINE HRESULT IDXGIDebug_QueryInterface(IDXGIDebug* This,REFIID riid,void **ppvObject) {
    return This->lpVtbl->QueryInterface(This,riid,ppvObject);
}
static __WIDL_INLINE ULONG IDXGIDebug_AddRef(IDXGIDebug* This) {
    return This->lpVtbl->AddRef(This);
}
static __WIDL_INLINE ULONG IDXGIDebug_Release(IDXGIDebug* This) {
    return This->lpVtbl->Release(This);
}
/*** IDXGIDebug methods ***/
static __WIDL_INLINE HRESULT IDXGIDebug_ReportLiveObjects(IDXGIDebug* This,GUID apiid,DXGI_DEBUG_RLO_FLAGS flags) {
    return This->lpVtbl->ReportLiveObjects(This,apiid,flags);
}
#endif
#endif

#endif


#endif  /* __IDXGIDebug_INTERFACE_DEFINED__ */

/*****************************************************************************
 * IDXGIDebug1 interface
 */
#ifndef __IDXGIDebug1_INTERFACE_DEFINED__
#define __IDXGIDebug1_INTERFACE_DEFINED__

DEFINE_GUID(IID_IDXGIDebug1, 0xc5a05f0c, 0x16f2, 0x4adf, 0x9f,0x4d, 0xa8,0xc4,0xd5,0x8a,0xc5,0x50);
#if defined(__cplusplus) && !defined(CINTERFACE)
MIDL_INTERFACE("c5a05f0c-16f2-4adf-9f4d-a8c4d58ac550")
IDXGIDebug1 : public IDXGIDebug
{
    virtual void STDMETHODCALLTYPE EnableLeakTrackingForThread(
        ) = 0;

    virtual void STDMETHODCALLTYPE DisableLeakTrackingForThread(
        ) = 0;

    virtual BOOL STDMETHODCALLTYPE IsLeakTrackingEnabledForThread(
        ) = 0;

};
#ifdef __CRT_UUID_DECL
__CRT_UUID_DECL(IDXGIDebug1, 0xc5a05f0c, 0x16f2, 0x4adf, 0x9f,0x4d, 0xa8,0xc4,0xd5,0x8a,0xc5,0x50)
#endif
#else
typedef struct IDXGIDebug1Vtbl {
    BEGIN_INTERFACE

    /*** IUnknown methods ***/
    HRESULT (STDMETHODCALLTYPE *QueryInterface)(
        IDXGIDebug1 *This,
        REFIID riid,
        void **ppvObject);

    ULONG (STDMETHODCALLTYPE *AddRef)(
        IDXGIDebug1 *This);

    ULONG (STDMETHODCALLTYPE *Release)(
        IDXGIDebug1 *This);

    /*** IDXGIDebug methods ***/
    HRESULT (STDMETHODCALLTYPE *ReportLiveObjects)(
        IDXGIDebug1 *This,
        GUID apiid,
        DXGI_DEBUG_RLO_FLAGS flags);

    /*** IDXGIDebug1 methods ***/
    void (STDMETHODCALLTYPE *EnableLeakTrackingForThread)(
        IDXGIDebug1 *This);

    void (STDMETHODCALLTYPE *DisableLeakTrackingForThread)(
        IDXGIDebug1 *This);

    BOOL (STDMETHODCALLTYPE *IsLeakTrackingEnabledForThread)(
        IDXGIDebug1 *This);

    END_INTERFACE
} IDXGIDebug1Vtbl;

interface IDXGIDebug1 {
    CONST_VTBL IDXGIDebug1Vtbl* lpVtbl;
};

#ifdef COBJMACROS
#ifndef WIDL_C_INLINE_WRAPPERS
/*** IUnknown methods ***/
#define IDXGIDebug1_QueryInterface(This,riid,ppvObject) (This)->lpVtbl->QueryInterface(This,riid,ppvObject)
#define IDXGIDebug1_AddRef(This) (This)->lpVtbl->AddRef(This)
#define IDXGIDebug1_Release(This) (This)->lpVtbl->Release(This)
/*** IDXGIDebug methods ***/
#define IDXGIDebug1_ReportLiveObjects(This,apiid,flags) (This)->lpVtbl->ReportLiveObjects(This,apiid,flags)
/*** IDXGIDebug1 methods ***/
#define IDXGIDebug1_EnableLeakTrackingForThread(This) (This)->lpVtbl->EnableLeakTrackingForThread(This)
#define IDXGIDebug1_DisableLeakTrackingForThread(This) (This)->lpVtbl->DisableLeakTrackingForThread(This)
#define IDXGIDebug1_IsLeakTrackingEnabledForThread(This) (This)->lpVtbl->IsLeakTrackingEnabledForThread(This)
#else
/*** IUnknown methods ***/
static __WIDL_INLINE HRESULT IDXGIDebug1_QueryInterface(IDXGIDebug1* This,REFIID riid,void **ppvObject) {
    return This->lpVtbl->QueryInterface(This,riid,ppvObject);
}
static __WIDL_INLINE ULONG IDXGIDebug1_AddRef(IDXGIDebug1* This) {
    return This->lpVtbl->AddRef(This);
}
static __WIDL_INLINE ULONG IDXGIDebug1_Release(IDXGIDebug1* This) {
    return This->lpVtbl->Release(This);
}
/*** IDXGIDebug methods ***/
static __WIDL_INLINE HRESULT IDXGIDebug1_ReportLiveObjects(IDXGIDebug1* This,GUID apiid,DXGI_DEBUG_RLO_FLAGS flags) {
    return This->lpVtbl->ReportLiveObjects(This,apiid,flags);
}
/*** IDXGIDebug1 methods ***/
static __WIDL_INLINE void IDXGIDebug1_EnableLeakTrackingForThread(IDXGIDebug1* This) {
    This->lpVtbl->EnableLeakTrackingForThread(This);
}
static __WIDL_INLINE void IDXGIDebug1_DisableLeakTrackingForThread(IDXGIDebug1* This) {
    This->lpVtbl->DisableLeakTrackingForThread(This);
}
static __WIDL_INLINE BOOL IDXGIDebug1_IsLeakTrackingEnabledForThread(IDXGIDebug1* This) {
    return This->lpVtbl->IsLeakTrackingEnabledForThread(This);
}
#endif
#endif

#endif


#endif  /* __IDXGIDebug1_INTERFACE_DEFINED__ */

/* Begin additional prototypes for all interfaces */


/* End additional prototypes */

#ifdef __cplusplus
}
#endif

#endif /* __dxgidebug_h__ */
