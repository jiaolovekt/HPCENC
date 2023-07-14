/*** Autogenerated by WIDL 8.0.1 from ../include/msdasc.idl - Do not edit ***/

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

#ifndef __msdasc_h__
#define __msdasc_h__

#ifndef __WIDL_INLINE
#if defined(__cplusplus) || defined(_MSC_VER)
#define __WIDL_INLINE inline
#elif defined(__GNUC__)
#define __WIDL_INLINE __inline__
#endif
#endif

/* Forward declarations */

#ifndef __IDataSourceLocator_FWD_DEFINED__
#define __IDataSourceLocator_FWD_DEFINED__
typedef interface IDataSourceLocator IDataSourceLocator;
#ifdef __cplusplus
interface IDataSourceLocator;
#endif /* __cplusplus */
#endif

#ifndef __IDBPromptInitialize_FWD_DEFINED__
#define __IDBPromptInitialize_FWD_DEFINED__
typedef interface IDBPromptInitialize IDBPromptInitialize;
#ifdef __cplusplus
interface IDBPromptInitialize;
#endif /* __cplusplus */
#endif

#ifndef __IDataInitialize_FWD_DEFINED__
#define __IDataInitialize_FWD_DEFINED__
typedef interface IDataInitialize IDataInitialize;
#ifdef __cplusplus
interface IDataInitialize;
#endif /* __cplusplus */
#endif

#ifndef __MSDAINITIALIZE_FWD_DEFINED__
#define __MSDAINITIALIZE_FWD_DEFINED__
#ifdef __cplusplus
typedef class MSDAINITIALIZE MSDAINITIALIZE;
#else
typedef struct MSDAINITIALIZE MSDAINITIALIZE;
#endif /* defined __cplusplus */
#endif /* defined __MSDAINITIALIZE_FWD_DEFINED__ */

#ifndef __DataLinks_FWD_DEFINED__
#define __DataLinks_FWD_DEFINED__
#ifdef __cplusplus
typedef class DataLinks DataLinks;
#else
typedef struct DataLinks DataLinks;
#endif /* defined __cplusplus */
#endif /* defined __DataLinks_FWD_DEFINED__ */

/* Headers for imported files */

#include <oaidl.h>
#include <ocidl.h>
#include <oledb.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef LONG_PTR COMPATIBLE_LONG;
#ifndef __MSDASC_LIBRARY_DEFINED__
#define __MSDASC_LIBRARY_DEFINED__

DEFINE_GUID(LIBID_MSDASC, 0x2206ceb0, 0x19c1, 0x11d1, 0x89,0xe0, 0x00,0xc0,0x4f,0xd7,0xa8,0x29);

typedef DWORD DBPROMPTOPTIONS;
typedef enum tagDBPROMPTOPTIONSENUM {
    DBPROMPTOPTIONS_NONE = 0x0,
    DBPROMPTOPTIONS_WIZARDSHEET = 0x1,
    DBPROMPTOPTIONS_PROPERTYSHEET = 0x2,
    DBPROMPTOPTIONS_BROWSEONLY = 0x8,
    DBPROMPTOPTIONS_DISABLE_PROVIDER_SELECTION = 0x10,
    DBPROMPTOPTIONS_DISABLESAVEPASSWORD = 0x20
} DBPROMPTOPTIONSENUM;
/*****************************************************************************
 * IDataSourceLocator interface
 */
#ifndef __IDataSourceLocator_INTERFACE_DEFINED__
#define __IDataSourceLocator_INTERFACE_DEFINED__

DEFINE_GUID(IID_IDataSourceLocator, 0x2206ccb2, 0x19c1, 0x11d1, 0x89,0xe0, 0x00,0xc0,0x4f,0xd7,0xa8,0x29);
#if defined(__cplusplus) && !defined(CINTERFACE)
MIDL_INTERFACE("2206ccb2-19c1-11d1-89e0-00c04fd7a829")
IDataSourceLocator : public IDispatch
{
    virtual HRESULT STDMETHODCALLTYPE get_hWnd(
        COMPATIBLE_LONG *phwndParent) = 0;

    virtual HRESULT STDMETHODCALLTYPE put_hWnd(
        COMPATIBLE_LONG hwndParent) = 0;

    virtual HRESULT STDMETHODCALLTYPE PromptNew(
        IDispatch **ppADOConnection) = 0;

    virtual HRESULT STDMETHODCALLTYPE PromptEdit(
        IDispatch **ppADOConnection,
        VARIANT_BOOL *pbSuccess) = 0;

};
#ifdef __CRT_UUID_DECL
__CRT_UUID_DECL(IDataSourceLocator, 0x2206ccb2, 0x19c1, 0x11d1, 0x89,0xe0, 0x00,0xc0,0x4f,0xd7,0xa8,0x29)
#endif
#else
typedef struct IDataSourceLocatorVtbl {
    BEGIN_INTERFACE

    /*** IUnknown methods ***/
    HRESULT (STDMETHODCALLTYPE *QueryInterface)(
        IDataSourceLocator *This,
        REFIID riid,
        void **ppvObject);

    ULONG (STDMETHODCALLTYPE *AddRef)(
        IDataSourceLocator *This);

    ULONG (STDMETHODCALLTYPE *Release)(
        IDataSourceLocator *This);

    /*** IDispatch methods ***/
    HRESULT (STDMETHODCALLTYPE *GetTypeInfoCount)(
        IDataSourceLocator *This,
        UINT *pctinfo);

    HRESULT (STDMETHODCALLTYPE *GetTypeInfo)(
        IDataSourceLocator *This,
        UINT iTInfo,
        LCID lcid,
        ITypeInfo **ppTInfo);

    HRESULT (STDMETHODCALLTYPE *GetIDsOfNames)(
        IDataSourceLocator *This,
        REFIID riid,
        LPOLESTR *rgszNames,
        UINT cNames,
        LCID lcid,
        DISPID *rgDispId);

    HRESULT (STDMETHODCALLTYPE *Invoke)(
        IDataSourceLocator *This,
        DISPID dispIdMember,
        REFIID riid,
        LCID lcid,
        WORD wFlags,
        DISPPARAMS *pDispParams,
        VARIANT *pVarResult,
        EXCEPINFO *pExcepInfo,
        UINT *puArgErr);

    /*** IDataSourceLocator methods ***/
    HRESULT (STDMETHODCALLTYPE *get_hWnd)(
        IDataSourceLocator *This,
        COMPATIBLE_LONG *phwndParent);

    HRESULT (STDMETHODCALLTYPE *put_hWnd)(
        IDataSourceLocator *This,
        COMPATIBLE_LONG hwndParent);

    HRESULT (STDMETHODCALLTYPE *PromptNew)(
        IDataSourceLocator *This,
        IDispatch **ppADOConnection);

    HRESULT (STDMETHODCALLTYPE *PromptEdit)(
        IDataSourceLocator *This,
        IDispatch **ppADOConnection,
        VARIANT_BOOL *pbSuccess);

    END_INTERFACE
} IDataSourceLocatorVtbl;

interface IDataSourceLocator {
    CONST_VTBL IDataSourceLocatorVtbl* lpVtbl;
};

#ifdef COBJMACROS
#ifndef WIDL_C_INLINE_WRAPPERS
/*** IUnknown methods ***/
#define IDataSourceLocator_QueryInterface(This,riid,ppvObject) (This)->lpVtbl->QueryInterface(This,riid,ppvObject)
#define IDataSourceLocator_AddRef(This) (This)->lpVtbl->AddRef(This)
#define IDataSourceLocator_Release(This) (This)->lpVtbl->Release(This)
/*** IDispatch methods ***/
#define IDataSourceLocator_GetTypeInfoCount(This,pctinfo) (This)->lpVtbl->GetTypeInfoCount(This,pctinfo)
#define IDataSourceLocator_GetTypeInfo(This,iTInfo,lcid,ppTInfo) (This)->lpVtbl->GetTypeInfo(This,iTInfo,lcid,ppTInfo)
#define IDataSourceLocator_GetIDsOfNames(This,riid,rgszNames,cNames,lcid,rgDispId) (This)->lpVtbl->GetIDsOfNames(This,riid,rgszNames,cNames,lcid,rgDispId)
#define IDataSourceLocator_Invoke(This,dispIdMember,riid,lcid,wFlags,pDispParams,pVarResult,pExcepInfo,puArgErr) (This)->lpVtbl->Invoke(This,dispIdMember,riid,lcid,wFlags,pDispParams,pVarResult,pExcepInfo,puArgErr)
/*** IDataSourceLocator methods ***/
#define IDataSourceLocator_get_hWnd(This,phwndParent) (This)->lpVtbl->get_hWnd(This,phwndParent)
#define IDataSourceLocator_put_hWnd(This,hwndParent) (This)->lpVtbl->put_hWnd(This,hwndParent)
#define IDataSourceLocator_PromptNew(This,ppADOConnection) (This)->lpVtbl->PromptNew(This,ppADOConnection)
#define IDataSourceLocator_PromptEdit(This,ppADOConnection,pbSuccess) (This)->lpVtbl->PromptEdit(This,ppADOConnection,pbSuccess)
#else
/*** IUnknown methods ***/
static __WIDL_INLINE HRESULT IDataSourceLocator_QueryInterface(IDataSourceLocator* This,REFIID riid,void **ppvObject) {
    return This->lpVtbl->QueryInterface(This,riid,ppvObject);
}
static __WIDL_INLINE ULONG IDataSourceLocator_AddRef(IDataSourceLocator* This) {
    return This->lpVtbl->AddRef(This);
}
static __WIDL_INLINE ULONG IDataSourceLocator_Release(IDataSourceLocator* This) {
    return This->lpVtbl->Release(This);
}
/*** IDispatch methods ***/
static __WIDL_INLINE HRESULT IDataSourceLocator_GetTypeInfoCount(IDataSourceLocator* This,UINT *pctinfo) {
    return This->lpVtbl->GetTypeInfoCount(This,pctinfo);
}
static __WIDL_INLINE HRESULT IDataSourceLocator_GetTypeInfo(IDataSourceLocator* This,UINT iTInfo,LCID lcid,ITypeInfo **ppTInfo) {
    return This->lpVtbl->GetTypeInfo(This,iTInfo,lcid,ppTInfo);
}
static __WIDL_INLINE HRESULT IDataSourceLocator_GetIDsOfNames(IDataSourceLocator* This,REFIID riid,LPOLESTR *rgszNames,UINT cNames,LCID lcid,DISPID *rgDispId) {
    return This->lpVtbl->GetIDsOfNames(This,riid,rgszNames,cNames,lcid,rgDispId);
}
static __WIDL_INLINE HRESULT IDataSourceLocator_Invoke(IDataSourceLocator* This,DISPID dispIdMember,REFIID riid,LCID lcid,WORD wFlags,DISPPARAMS *pDispParams,VARIANT *pVarResult,EXCEPINFO *pExcepInfo,UINT *puArgErr) {
    return This->lpVtbl->Invoke(This,dispIdMember,riid,lcid,wFlags,pDispParams,pVarResult,pExcepInfo,puArgErr);
}
/*** IDataSourceLocator methods ***/
static __WIDL_INLINE HRESULT IDataSourceLocator_get_hWnd(IDataSourceLocator* This,COMPATIBLE_LONG *phwndParent) {
    return This->lpVtbl->get_hWnd(This,phwndParent);
}
static __WIDL_INLINE HRESULT IDataSourceLocator_put_hWnd(IDataSourceLocator* This,COMPATIBLE_LONG hwndParent) {
    return This->lpVtbl->put_hWnd(This,hwndParent);
}
static __WIDL_INLINE HRESULT IDataSourceLocator_PromptNew(IDataSourceLocator* This,IDispatch **ppADOConnection) {
    return This->lpVtbl->PromptNew(This,ppADOConnection);
}
static __WIDL_INLINE HRESULT IDataSourceLocator_PromptEdit(IDataSourceLocator* This,IDispatch **ppADOConnection,VARIANT_BOOL *pbSuccess) {
    return This->lpVtbl->PromptEdit(This,ppADOConnection,pbSuccess);
}
#endif
#endif

#endif


#endif  /* __IDataSourceLocator_INTERFACE_DEFINED__ */

/*****************************************************************************
 * IDBPromptInitialize interface
 */
#ifndef __IDBPromptInitialize_INTERFACE_DEFINED__
#define __IDBPromptInitialize_INTERFACE_DEFINED__

DEFINE_GUID(IID_IDBPromptInitialize, 0x2206ccb0, 0x19c1, 0x11d1, 0x89,0xe0, 0x00,0xc0,0x4f,0xd7,0xa8,0x29);
#if defined(__cplusplus) && !defined(CINTERFACE)
MIDL_INTERFACE("2206ccb0-19c1-11d1-89e0-00c04fd7a829")
IDBPromptInitialize : public IUnknown
{
    virtual HRESULT __stdcall PromptDataSource(
        IUnknown *pUnkOuter,
        HWND hWndParent,
        DBPROMPTOPTIONS dwPromptOptions,
        ULONG cSourceTypeFilter,
        DBSOURCETYPE *rgSourceTypeFilter,
        LPWSTR pwszszzProviderFilter,
        GUID *riid,
        IUnknown **ppDataSource) = 0;

    virtual HRESULT __stdcall PromptFileName(
        HWND hWndParent,
        ULONG dwPromptOptions,
        LPWSTR pwszInitialDirectory,
        LPWSTR pwszInitialFile,
        LPWSTR *ppwszSelectedFile) = 0;

};
#ifdef __CRT_UUID_DECL
__CRT_UUID_DECL(IDBPromptInitialize, 0x2206ccb0, 0x19c1, 0x11d1, 0x89,0xe0, 0x00,0xc0,0x4f,0xd7,0xa8,0x29)
#endif
#else
typedef struct IDBPromptInitializeVtbl {
    BEGIN_INTERFACE

    /*** IUnknown methods ***/
    HRESULT (STDMETHODCALLTYPE *QueryInterface)(
        IDBPromptInitialize *This,
        REFIID riid,
        void **ppvObject);

    ULONG (STDMETHODCALLTYPE *AddRef)(
        IDBPromptInitialize *This);

    ULONG (STDMETHODCALLTYPE *Release)(
        IDBPromptInitialize *This);

    /*** IDBPromptInitialize methods ***/
    HRESULT (__stdcall *PromptDataSource)(
        IDBPromptInitialize *This,
        IUnknown *pUnkOuter,
        HWND hWndParent,
        DBPROMPTOPTIONS dwPromptOptions,
        ULONG cSourceTypeFilter,
        DBSOURCETYPE *rgSourceTypeFilter,
        LPWSTR pwszszzProviderFilter,
        GUID *riid,
        IUnknown **ppDataSource);

    HRESULT (__stdcall *PromptFileName)(
        IDBPromptInitialize *This,
        HWND hWndParent,
        ULONG dwPromptOptions,
        LPWSTR pwszInitialDirectory,
        LPWSTR pwszInitialFile,
        LPWSTR *ppwszSelectedFile);

    END_INTERFACE
} IDBPromptInitializeVtbl;

interface IDBPromptInitialize {
    CONST_VTBL IDBPromptInitializeVtbl* lpVtbl;
};

#ifdef COBJMACROS
#ifndef WIDL_C_INLINE_WRAPPERS
/*** IUnknown methods ***/
#define IDBPromptInitialize_QueryInterface(This,riid,ppvObject) (This)->lpVtbl->QueryInterface(This,riid,ppvObject)
#define IDBPromptInitialize_AddRef(This) (This)->lpVtbl->AddRef(This)
#define IDBPromptInitialize_Release(This) (This)->lpVtbl->Release(This)
/*** IDBPromptInitialize methods ***/
#define IDBPromptInitialize_PromptDataSource(This,pUnkOuter,hWndParent,dwPromptOptions,cSourceTypeFilter,rgSourceTypeFilter,pwszszzProviderFilter,riid,ppDataSource) (This)->lpVtbl->PromptDataSource(This,pUnkOuter,hWndParent,dwPromptOptions,cSourceTypeFilter,rgSourceTypeFilter,pwszszzProviderFilter,riid,ppDataSource)
#define IDBPromptInitialize_PromptFileName(This,hWndParent,dwPromptOptions,pwszInitialDirectory,pwszInitialFile,ppwszSelectedFile) (This)->lpVtbl->PromptFileName(This,hWndParent,dwPromptOptions,pwszInitialDirectory,pwszInitialFile,ppwszSelectedFile)
#else
/*** IUnknown methods ***/
static __WIDL_INLINE HRESULT IDBPromptInitialize_QueryInterface(IDBPromptInitialize* This,REFIID riid,void **ppvObject) {
    return This->lpVtbl->QueryInterface(This,riid,ppvObject);
}
static __WIDL_INLINE ULONG IDBPromptInitialize_AddRef(IDBPromptInitialize* This) {
    return This->lpVtbl->AddRef(This);
}
static __WIDL_INLINE ULONG IDBPromptInitialize_Release(IDBPromptInitialize* This) {
    return This->lpVtbl->Release(This);
}
/*** IDBPromptInitialize methods ***/
static __WIDL_INLINE HRESULT IDBPromptInitialize_PromptDataSource(IDBPromptInitialize* This,IUnknown *pUnkOuter,HWND hWndParent,DBPROMPTOPTIONS dwPromptOptions,ULONG cSourceTypeFilter,DBSOURCETYPE *rgSourceTypeFilter,LPWSTR pwszszzProviderFilter,GUID *riid,IUnknown **ppDataSource) {
    return This->lpVtbl->PromptDataSource(This,pUnkOuter,hWndParent,dwPromptOptions,cSourceTypeFilter,rgSourceTypeFilter,pwszszzProviderFilter,riid,ppDataSource);
}
static __WIDL_INLINE HRESULT IDBPromptInitialize_PromptFileName(IDBPromptInitialize* This,HWND hWndParent,ULONG dwPromptOptions,LPWSTR pwszInitialDirectory,LPWSTR pwszInitialFile,LPWSTR *ppwszSelectedFile) {
    return This->lpVtbl->PromptFileName(This,hWndParent,dwPromptOptions,pwszInitialDirectory,pwszInitialFile,ppwszSelectedFile);
}
#endif
#endif

#endif


#endif  /* __IDBPromptInitialize_INTERFACE_DEFINED__ */

/*****************************************************************************
 * IDataInitialize interface
 */
#ifndef __IDataInitialize_INTERFACE_DEFINED__
#define __IDataInitialize_INTERFACE_DEFINED__

DEFINE_GUID(IID_IDataInitialize, 0x2206ccb1, 0x19c1, 0x11d1, 0x89,0xe0, 0x00,0xc0,0x4f,0xd7,0xa8,0x29);
#if defined(__cplusplus) && !defined(CINTERFACE)
MIDL_INTERFACE("2206ccb1-19c1-11d1-89e0-00c04fd7a829")
IDataInitialize : public IUnknown
{
    virtual HRESULT STDMETHODCALLTYPE GetDataSource(
        IUnknown *pUnkOuter,
        DWORD dwClsCtx,
        LPWSTR pwszInitializationString,
        REFIID riid,
        IUnknown **ppDataSource) = 0;

    virtual HRESULT STDMETHODCALLTYPE GetInitializationString(
        IUnknown *pDataSource,
        boolean fIncludePassword,
        LPWSTR *ppwszInitString) = 0;

    virtual HRESULT STDMETHODCALLTYPE CreateDBInstance(
        REFCLSID clsidProvider,
        IUnknown *pUnkOuter,
        DWORD dwClsCtx,
        LPWSTR pwszReserved,
        REFIID riid,
        IUnknown **ppDataSource) = 0;

    virtual HRESULT STDMETHODCALLTYPE CreateDBInstanceEx(
        REFCLSID clsidProvider,
        IUnknown *pUnkOuter,
        DWORD dwClsCtx,
        LPWSTR pwszReserved,
        COSERVERINFO *pServerInfo,
        DWORD cmq,
        MULTI_QI *results) = 0;

    virtual HRESULT STDMETHODCALLTYPE LoadStringFromStorage(
        LPWSTR pwszFileName,
        LPWSTR *ppwszInitializationString) = 0;

    virtual HRESULT STDMETHODCALLTYPE WriteStringToStorage(
        LPWSTR pwszFileName,
        LPWSTR pwszInitializationString,
        DWORD dwCreationDisposition) = 0;

};
#ifdef __CRT_UUID_DECL
__CRT_UUID_DECL(IDataInitialize, 0x2206ccb1, 0x19c1, 0x11d1, 0x89,0xe0, 0x00,0xc0,0x4f,0xd7,0xa8,0x29)
#endif
#else
typedef struct IDataInitializeVtbl {
    BEGIN_INTERFACE

    /*** IUnknown methods ***/
    HRESULT (STDMETHODCALLTYPE *QueryInterface)(
        IDataInitialize *This,
        REFIID riid,
        void **ppvObject);

    ULONG (STDMETHODCALLTYPE *AddRef)(
        IDataInitialize *This);

    ULONG (STDMETHODCALLTYPE *Release)(
        IDataInitialize *This);

    /*** IDataInitialize methods ***/
    HRESULT (STDMETHODCALLTYPE *GetDataSource)(
        IDataInitialize *This,
        IUnknown *pUnkOuter,
        DWORD dwClsCtx,
        LPWSTR pwszInitializationString,
        REFIID riid,
        IUnknown **ppDataSource);

    HRESULT (STDMETHODCALLTYPE *GetInitializationString)(
        IDataInitialize *This,
        IUnknown *pDataSource,
        boolean fIncludePassword,
        LPWSTR *ppwszInitString);

    HRESULT (STDMETHODCALLTYPE *CreateDBInstance)(
        IDataInitialize *This,
        REFCLSID clsidProvider,
        IUnknown *pUnkOuter,
        DWORD dwClsCtx,
        LPWSTR pwszReserved,
        REFIID riid,
        IUnknown **ppDataSource);

    HRESULT (STDMETHODCALLTYPE *CreateDBInstanceEx)(
        IDataInitialize *This,
        REFCLSID clsidProvider,
        IUnknown *pUnkOuter,
        DWORD dwClsCtx,
        LPWSTR pwszReserved,
        COSERVERINFO *pServerInfo,
        DWORD cmq,
        MULTI_QI *results);

    HRESULT (STDMETHODCALLTYPE *LoadStringFromStorage)(
        IDataInitialize *This,
        LPWSTR pwszFileName,
        LPWSTR *ppwszInitializationString);

    HRESULT (STDMETHODCALLTYPE *WriteStringToStorage)(
        IDataInitialize *This,
        LPWSTR pwszFileName,
        LPWSTR pwszInitializationString,
        DWORD dwCreationDisposition);

    END_INTERFACE
} IDataInitializeVtbl;

interface IDataInitialize {
    CONST_VTBL IDataInitializeVtbl* lpVtbl;
};

#ifdef COBJMACROS
#ifndef WIDL_C_INLINE_WRAPPERS
/*** IUnknown methods ***/
#define IDataInitialize_QueryInterface(This,riid,ppvObject) (This)->lpVtbl->QueryInterface(This,riid,ppvObject)
#define IDataInitialize_AddRef(This) (This)->lpVtbl->AddRef(This)
#define IDataInitialize_Release(This) (This)->lpVtbl->Release(This)
/*** IDataInitialize methods ***/
#define IDataInitialize_GetDataSource(This,pUnkOuter,dwClsCtx,pwszInitializationString,riid,ppDataSource) (This)->lpVtbl->GetDataSource(This,pUnkOuter,dwClsCtx,pwszInitializationString,riid,ppDataSource)
#define IDataInitialize_GetInitializationString(This,pDataSource,fIncludePassword,ppwszInitString) (This)->lpVtbl->GetInitializationString(This,pDataSource,fIncludePassword,ppwszInitString)
#define IDataInitialize_CreateDBInstance(This,clsidProvider,pUnkOuter,dwClsCtx,pwszReserved,riid,ppDataSource) (This)->lpVtbl->CreateDBInstance(This,clsidProvider,pUnkOuter,dwClsCtx,pwszReserved,riid,ppDataSource)
#define IDataInitialize_CreateDBInstanceEx(This,clsidProvider,pUnkOuter,dwClsCtx,pwszReserved,pServerInfo,cmq,results) (This)->lpVtbl->CreateDBInstanceEx(This,clsidProvider,pUnkOuter,dwClsCtx,pwszReserved,pServerInfo,cmq,results)
#define IDataInitialize_LoadStringFromStorage(This,pwszFileName,ppwszInitializationString) (This)->lpVtbl->LoadStringFromStorage(This,pwszFileName,ppwszInitializationString)
#define IDataInitialize_WriteStringToStorage(This,pwszFileName,pwszInitializationString,dwCreationDisposition) (This)->lpVtbl->WriteStringToStorage(This,pwszFileName,pwszInitializationString,dwCreationDisposition)
#else
/*** IUnknown methods ***/
static __WIDL_INLINE HRESULT IDataInitialize_QueryInterface(IDataInitialize* This,REFIID riid,void **ppvObject) {
    return This->lpVtbl->QueryInterface(This,riid,ppvObject);
}
static __WIDL_INLINE ULONG IDataInitialize_AddRef(IDataInitialize* This) {
    return This->lpVtbl->AddRef(This);
}
static __WIDL_INLINE ULONG IDataInitialize_Release(IDataInitialize* This) {
    return This->lpVtbl->Release(This);
}
/*** IDataInitialize methods ***/
static __WIDL_INLINE HRESULT IDataInitialize_GetDataSource(IDataInitialize* This,IUnknown *pUnkOuter,DWORD dwClsCtx,LPWSTR pwszInitializationString,REFIID riid,IUnknown **ppDataSource) {
    return This->lpVtbl->GetDataSource(This,pUnkOuter,dwClsCtx,pwszInitializationString,riid,ppDataSource);
}
static __WIDL_INLINE HRESULT IDataInitialize_GetInitializationString(IDataInitialize* This,IUnknown *pDataSource,boolean fIncludePassword,LPWSTR *ppwszInitString) {
    return This->lpVtbl->GetInitializationString(This,pDataSource,fIncludePassword,ppwszInitString);
}
static __WIDL_INLINE HRESULT IDataInitialize_CreateDBInstance(IDataInitialize* This,REFCLSID clsidProvider,IUnknown *pUnkOuter,DWORD dwClsCtx,LPWSTR pwszReserved,REFIID riid,IUnknown **ppDataSource) {
    return This->lpVtbl->CreateDBInstance(This,clsidProvider,pUnkOuter,dwClsCtx,pwszReserved,riid,ppDataSource);
}
static __WIDL_INLINE HRESULT IDataInitialize_CreateDBInstanceEx(IDataInitialize* This,REFCLSID clsidProvider,IUnknown *pUnkOuter,DWORD dwClsCtx,LPWSTR pwszReserved,COSERVERINFO *pServerInfo,DWORD cmq,MULTI_QI *results) {
    return This->lpVtbl->CreateDBInstanceEx(This,clsidProvider,pUnkOuter,dwClsCtx,pwszReserved,pServerInfo,cmq,results);
}
static __WIDL_INLINE HRESULT IDataInitialize_LoadStringFromStorage(IDataInitialize* This,LPWSTR pwszFileName,LPWSTR *ppwszInitializationString) {
    return This->lpVtbl->LoadStringFromStorage(This,pwszFileName,ppwszInitializationString);
}
static __WIDL_INLINE HRESULT IDataInitialize_WriteStringToStorage(IDataInitialize* This,LPWSTR pwszFileName,LPWSTR pwszInitializationString,DWORD dwCreationDisposition) {
    return This->lpVtbl->WriteStringToStorage(This,pwszFileName,pwszInitializationString,dwCreationDisposition);
}
#endif
#endif

#endif

HRESULT STDMETHODCALLTYPE IDataInitialize_RemoteCreateDBInstanceEx_Proxy(
    IDataInitialize* This,
    REFCLSID clsidProvider,
    IUnknown *pUnkOuter,
    DWORD dwClsCtx,
    LPWSTR pwszReserved,
    COSERVERINFO *pServerInfo,
    DWORD cmq,
    const IID **iids,
    IUnknown **ifs,
    HRESULT *hr);
void __RPC_STUB IDataInitialize_RemoteCreateDBInstanceEx_Stub(
    IRpcStubBuffer* This,
    IRpcChannelBuffer* pRpcChannelBuffer,
    PRPC_MESSAGE pRpcMessage,
    DWORD* pdwStubPhase);
HRESULT CALLBACK IDataInitialize_CreateDBInstanceEx_Proxy(
    IDataInitialize* This,
    REFCLSID clsidProvider,
    IUnknown *pUnkOuter,
    DWORD dwClsCtx,
    LPWSTR pwszReserved,
    COSERVERINFO *pServerInfo,
    DWORD cmq,
    MULTI_QI *results);
HRESULT __RPC_STUB IDataInitialize_CreateDBInstanceEx_Stub(
    IDataInitialize* This,
    REFCLSID clsidProvider,
    IUnknown *pUnkOuter,
    DWORD dwClsCtx,
    LPWSTR pwszReserved,
    COSERVERINFO *pServerInfo,
    DWORD cmq,
    const IID **iids,
    IUnknown **ifs,
    HRESULT *hr);

#endif  /* __IDataInitialize_INTERFACE_DEFINED__ */

/*****************************************************************************
 * MSDAINITIALIZE coclass
 */

DEFINE_GUID(CLSID_MSDAINITIALIZE, 0x2206cdb0, 0x19c1, 0x11d1, 0x89,0xe0, 0x00,0xc0,0x4f,0xd7,0xa8,0x29);

#ifdef __cplusplus
class DECLSPEC_UUID("2206cdb0-19c1-11d1-89e0-00c04fd7a829") MSDAINITIALIZE;
#ifdef __CRT_UUID_DECL
__CRT_UUID_DECL(MSDAINITIALIZE, 0x2206cdb0, 0x19c1, 0x11d1, 0x89,0xe0, 0x00,0xc0,0x4f,0xd7,0xa8,0x29)
#endif
#endif

/*****************************************************************************
 * DataLinks coclass
 */

DEFINE_GUID(CLSID_DataLinks, 0x2206cdb2, 0x19c1, 0x11d1, 0x89,0xe0, 0x00,0xc0,0x4f,0xd7,0xa8,0x29);

#ifdef __cplusplus
class DECLSPEC_UUID("2206cdb2-19c1-11d1-89e0-00c04fd7a829") DataLinks;
#ifdef __CRT_UUID_DECL
__CRT_UUID_DECL(DataLinks, 0x2206cdb2, 0x19c1, 0x11d1, 0x89,0xe0, 0x00,0xc0,0x4f,0xd7,0xa8,0x29)
#endif
#endif

#endif /* __MSDASC_LIBRARY_DEFINED__ */
/* Begin additional prototypes for all interfaces */

ULONG           __RPC_USER HWND_UserSize     (ULONG *, ULONG, HWND *);
unsigned char * __RPC_USER HWND_UserMarshal  (ULONG *, unsigned char *, HWND *);
unsigned char * __RPC_USER HWND_UserUnmarshal(ULONG *, unsigned char *, HWND *);
void            __RPC_USER HWND_UserFree     (ULONG *, HWND *);

/* End additional prototypes */

#ifdef __cplusplus
}
#endif

#endif /* __msdasc_h__ */
