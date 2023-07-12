/*** Autogenerated by WIDL 8.0.1 from ../include/wmiutils.idl - Do not edit ***/

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

#ifndef __wmiutils_h__
#define __wmiutils_h__

#ifndef __WIDL_INLINE
#if defined(__cplusplus) || defined(_MSC_VER)
#define __WIDL_INLINE inline
#elif defined(__GNUC__)
#define __WIDL_INLINE __inline__
#endif
#endif

/* Forward declarations */

#ifndef __IWbemPathKeyList_FWD_DEFINED__
#define __IWbemPathKeyList_FWD_DEFINED__
typedef interface IWbemPathKeyList IWbemPathKeyList;
#ifdef __cplusplus
interface IWbemPathKeyList;
#endif /* __cplusplus */
#endif

#ifndef __IWbemPath_FWD_DEFINED__
#define __IWbemPath_FWD_DEFINED__
typedef interface IWbemPath IWbemPath;
#ifdef __cplusplus
interface IWbemPath;
#endif /* __cplusplus */
#endif

#ifndef __WbemDefPath_FWD_DEFINED__
#define __WbemDefPath_FWD_DEFINED__
#ifdef __cplusplus
typedef class WbemDefPath WbemDefPath;
#else
typedef struct WbemDefPath WbemDefPath;
#endif /* defined __cplusplus */
#endif /* defined __WbemDefPath_FWD_DEFINED__ */

/* Headers for imported files */

#include <oaidl.h>

#ifdef __cplusplus
extern "C" {
#endif

#ifndef __IWbemPath_FWD_DEFINED__
#define __IWbemPath_FWD_DEFINED__
typedef interface IWbemPath IWbemPath;
#ifdef __cplusplus
interface IWbemPath;
#endif /* __cplusplus */
#endif

#ifndef __IWbemPathKeyList_FWD_DEFINED__
#define __IWbemPathKeyList_FWD_DEFINED__
typedef interface IWbemPathKeyList IWbemPathKeyList;
#ifdef __cplusplus
interface IWbemPathKeyList;
#endif /* __cplusplus */
#endif

typedef enum tag_WBEM_PATH_STATUS_FLAG {
    WBEMPATH_INFO_ANON_LOCAL_MACHINE = 0x1,
    WBEMPATH_INFO_HAS_MACHINE_NAME = 0x2,
    WBEMPATH_INFO_IS_CLASS_REF = 0x4,
    WBEMPATH_INFO_IS_INST_REF = 0x8,
    WBEMPATH_INFO_HAS_SUBSCOPES = 0x10,
    WBEMPATH_INFO_IS_COMPOUND = 0x20,
    WBEMPATH_INFO_HAS_V2_REF_PATHS = 0x40,
    WBEMPATH_INFO_HAS_IMPLIED_KEY = 0x80,
    WBEMPATH_INFO_CONTAINS_SINGLETON = 0x100,
    WBEMPATH_INFO_V1_COMPLIANT = 0x200,
    WBEMPATH_INFO_V2_COMPLIANT = 0x400,
    WBEMPATH_INFO_CIM_COMPLIANT = 0x800,
    WBEMPATH_INFO_IS_SINGLETON = 0x1000,
    WBEMPATH_INFO_IS_PARENT = 0x2000,
    WBEMPATH_INFO_SERVER_NAMESPACE_ONLY = 0x4000,
    WBEMPATH_INFO_NATIVE_PATH = 0x8000,
    WBEMPATH_INFO_WMI_PATH = 0x10000,
    WBEMPATH_INFO_PATH_HAD_SERVER = 0x20000
} tag_WBEM_PATH_STATUS_FLAG;
typedef enum tag_WBEM_PATH_CREATE_FLAG {
    WBEMPATH_CREATE_ACCEPT_RELATIVE = 0x1,
    WBEMPATH_CREATE_ACCEPT_ABSOLUTE = 0x2,
    WBEMPATH_CREATE_ACCEPT_ALL = 0x4,
    WBEMPATH_TREAT_SINGLE_IDENT_AS_NS = 0x8
} tag_WBEM_PATH_CREATE_FLAG;
typedef enum tag_WBEM_GET_TEXT_FLAGS {
    WBEMPATH_COMPRESSED = 0x1,
    WBEMPATH_GET_RELATIVE_ONLY = 0x2,
    WBEMPATH_GET_SERVER_TOO = 0x4,
    WBEMPATH_GET_SERVER_AND_NAMESPACE_ONLY = 0x8,
    WBEMPATH_GET_NAMESPACE_ONLY = 0x10,
    WBEMPATH_GET_ORIGINAL = 0x20
} tag_WBEM_GET_TEXT_FLAGS;
/*****************************************************************************
 * IWbemPathKeyList interface
 */
#ifndef __IWbemPathKeyList_INTERFACE_DEFINED__
#define __IWbemPathKeyList_INTERFACE_DEFINED__

DEFINE_GUID(IID_IWbemPathKeyList, 0x9ae62877, 0x7544, 0x4bb0, 0xaa,0x26, 0xa1,0x38,0x24,0x65,0x9e,0xd6);
#if defined(__cplusplus) && !defined(CINTERFACE)
MIDL_INTERFACE("9ae62877-7544-4bb0-aa26-a13824659ed6")
IWbemPathKeyList : public IUnknown
{
    virtual HRESULT STDMETHODCALLTYPE GetCount(
        ULONG *puKeyCount) = 0;

    virtual HRESULT STDMETHODCALLTYPE SetKey(
        LPCWSTR wszName,
        ULONG uFlags,
        ULONG uCimType,
        LPVOID pKeyVal) = 0;

    virtual HRESULT STDMETHODCALLTYPE SetKey2(
        LPCWSTR wszName,
        ULONG uFlags,
        ULONG uCimType,
        VARIANT *pKeyVal) = 0;

    virtual HRESULT STDMETHODCALLTYPE GetKey(
        ULONG uKeyIx,
        ULONG uFlags,
        ULONG *puNameBufSize,
        LPWSTR pszKeyName,
        ULONG *puKeyValBufSize,
        LPVOID pKeyVal,
        ULONG *puApparentCimType) = 0;

    virtual HRESULT STDMETHODCALLTYPE GetKey2(
        ULONG uKeyIx,
        ULONG uFlags,
        ULONG *puNameBufSize,
        LPWSTR pszKeyName,
        VARIANT *pKeyValue,
        ULONG *puApparentCimType) = 0;

    virtual HRESULT STDMETHODCALLTYPE RemoveKey(
        LPCWSTR wszName,
        ULONG uFlags) = 0;

    virtual HRESULT STDMETHODCALLTYPE RemoveAllKeys(
        ULONG uFlags) = 0;

    virtual HRESULT STDMETHODCALLTYPE MakeSingleton(
        boolean bSet) = 0;

    virtual HRESULT STDMETHODCALLTYPE GetInfo(
        ULONG uRequestedInfo,
        ULONGLONG *puResponse) = 0;

    virtual HRESULT STDMETHODCALLTYPE GetText(
        LONG lFlags,
        ULONG *puBuffLength,
        LPWSTR pszText) = 0;

};
#ifdef __CRT_UUID_DECL
__CRT_UUID_DECL(IWbemPathKeyList, 0x9ae62877, 0x7544, 0x4bb0, 0xaa,0x26, 0xa1,0x38,0x24,0x65,0x9e,0xd6)
#endif
#else
typedef struct IWbemPathKeyListVtbl {
    BEGIN_INTERFACE

    /*** IUnknown methods ***/
    HRESULT (STDMETHODCALLTYPE *QueryInterface)(
        IWbemPathKeyList *This,
        REFIID riid,
        void **ppvObject);

    ULONG (STDMETHODCALLTYPE *AddRef)(
        IWbemPathKeyList *This);

    ULONG (STDMETHODCALLTYPE *Release)(
        IWbemPathKeyList *This);

    /*** IWbemPathKeyList methods ***/
    HRESULT (STDMETHODCALLTYPE *GetCount)(
        IWbemPathKeyList *This,
        ULONG *puKeyCount);

    HRESULT (STDMETHODCALLTYPE *SetKey)(
        IWbemPathKeyList *This,
        LPCWSTR wszName,
        ULONG uFlags,
        ULONG uCimType,
        LPVOID pKeyVal);

    HRESULT (STDMETHODCALLTYPE *SetKey2)(
        IWbemPathKeyList *This,
        LPCWSTR wszName,
        ULONG uFlags,
        ULONG uCimType,
        VARIANT *pKeyVal);

    HRESULT (STDMETHODCALLTYPE *GetKey)(
        IWbemPathKeyList *This,
        ULONG uKeyIx,
        ULONG uFlags,
        ULONG *puNameBufSize,
        LPWSTR pszKeyName,
        ULONG *puKeyValBufSize,
        LPVOID pKeyVal,
        ULONG *puApparentCimType);

    HRESULT (STDMETHODCALLTYPE *GetKey2)(
        IWbemPathKeyList *This,
        ULONG uKeyIx,
        ULONG uFlags,
        ULONG *puNameBufSize,
        LPWSTR pszKeyName,
        VARIANT *pKeyValue,
        ULONG *puApparentCimType);

    HRESULT (STDMETHODCALLTYPE *RemoveKey)(
        IWbemPathKeyList *This,
        LPCWSTR wszName,
        ULONG uFlags);

    HRESULT (STDMETHODCALLTYPE *RemoveAllKeys)(
        IWbemPathKeyList *This,
        ULONG uFlags);

    HRESULT (STDMETHODCALLTYPE *MakeSingleton)(
        IWbemPathKeyList *This,
        boolean bSet);

    HRESULT (STDMETHODCALLTYPE *GetInfo)(
        IWbemPathKeyList *This,
        ULONG uRequestedInfo,
        ULONGLONG *puResponse);

    HRESULT (STDMETHODCALLTYPE *GetText)(
        IWbemPathKeyList *This,
        LONG lFlags,
        ULONG *puBuffLength,
        LPWSTR pszText);

    END_INTERFACE
} IWbemPathKeyListVtbl;

interface IWbemPathKeyList {
    CONST_VTBL IWbemPathKeyListVtbl* lpVtbl;
};

#ifdef COBJMACROS
#ifndef WIDL_C_INLINE_WRAPPERS
/*** IUnknown methods ***/
#define IWbemPathKeyList_QueryInterface(This,riid,ppvObject) (This)->lpVtbl->QueryInterface(This,riid,ppvObject)
#define IWbemPathKeyList_AddRef(This) (This)->lpVtbl->AddRef(This)
#define IWbemPathKeyList_Release(This) (This)->lpVtbl->Release(This)
/*** IWbemPathKeyList methods ***/
#define IWbemPathKeyList_GetCount(This,puKeyCount) (This)->lpVtbl->GetCount(This,puKeyCount)
#define IWbemPathKeyList_SetKey(This,wszName,uFlags,uCimType,pKeyVal) (This)->lpVtbl->SetKey(This,wszName,uFlags,uCimType,pKeyVal)
#define IWbemPathKeyList_SetKey2(This,wszName,uFlags,uCimType,pKeyVal) (This)->lpVtbl->SetKey2(This,wszName,uFlags,uCimType,pKeyVal)
#define IWbemPathKeyList_GetKey(This,uKeyIx,uFlags,puNameBufSize,pszKeyName,puKeyValBufSize,pKeyVal,puApparentCimType) (This)->lpVtbl->GetKey(This,uKeyIx,uFlags,puNameBufSize,pszKeyName,puKeyValBufSize,pKeyVal,puApparentCimType)
#define IWbemPathKeyList_GetKey2(This,uKeyIx,uFlags,puNameBufSize,pszKeyName,pKeyValue,puApparentCimType) (This)->lpVtbl->GetKey2(This,uKeyIx,uFlags,puNameBufSize,pszKeyName,pKeyValue,puApparentCimType)
#define IWbemPathKeyList_RemoveKey(This,wszName,uFlags) (This)->lpVtbl->RemoveKey(This,wszName,uFlags)
#define IWbemPathKeyList_RemoveAllKeys(This,uFlags) (This)->lpVtbl->RemoveAllKeys(This,uFlags)
#define IWbemPathKeyList_MakeSingleton(This,bSet) (This)->lpVtbl->MakeSingleton(This,bSet)
#define IWbemPathKeyList_GetInfo(This,uRequestedInfo,puResponse) (This)->lpVtbl->GetInfo(This,uRequestedInfo,puResponse)
#define IWbemPathKeyList_GetText(This,lFlags,puBuffLength,pszText) (This)->lpVtbl->GetText(This,lFlags,puBuffLength,pszText)
#else
/*** IUnknown methods ***/
static __WIDL_INLINE HRESULT IWbemPathKeyList_QueryInterface(IWbemPathKeyList* This,REFIID riid,void **ppvObject) {
    return This->lpVtbl->QueryInterface(This,riid,ppvObject);
}
static __WIDL_INLINE ULONG IWbemPathKeyList_AddRef(IWbemPathKeyList* This) {
    return This->lpVtbl->AddRef(This);
}
static __WIDL_INLINE ULONG IWbemPathKeyList_Release(IWbemPathKeyList* This) {
    return This->lpVtbl->Release(This);
}
/*** IWbemPathKeyList methods ***/
static __WIDL_INLINE HRESULT IWbemPathKeyList_GetCount(IWbemPathKeyList* This,ULONG *puKeyCount) {
    return This->lpVtbl->GetCount(This,puKeyCount);
}
static __WIDL_INLINE HRESULT IWbemPathKeyList_SetKey(IWbemPathKeyList* This,LPCWSTR wszName,ULONG uFlags,ULONG uCimType,LPVOID pKeyVal) {
    return This->lpVtbl->SetKey(This,wszName,uFlags,uCimType,pKeyVal);
}
static __WIDL_INLINE HRESULT IWbemPathKeyList_SetKey2(IWbemPathKeyList* This,LPCWSTR wszName,ULONG uFlags,ULONG uCimType,VARIANT *pKeyVal) {
    return This->lpVtbl->SetKey2(This,wszName,uFlags,uCimType,pKeyVal);
}
static __WIDL_INLINE HRESULT IWbemPathKeyList_GetKey(IWbemPathKeyList* This,ULONG uKeyIx,ULONG uFlags,ULONG *puNameBufSize,LPWSTR pszKeyName,ULONG *puKeyValBufSize,LPVOID pKeyVal,ULONG *puApparentCimType) {
    return This->lpVtbl->GetKey(This,uKeyIx,uFlags,puNameBufSize,pszKeyName,puKeyValBufSize,pKeyVal,puApparentCimType);
}
static __WIDL_INLINE HRESULT IWbemPathKeyList_GetKey2(IWbemPathKeyList* This,ULONG uKeyIx,ULONG uFlags,ULONG *puNameBufSize,LPWSTR pszKeyName,VARIANT *pKeyValue,ULONG *puApparentCimType) {
    return This->lpVtbl->GetKey2(This,uKeyIx,uFlags,puNameBufSize,pszKeyName,pKeyValue,puApparentCimType);
}
static __WIDL_INLINE HRESULT IWbemPathKeyList_RemoveKey(IWbemPathKeyList* This,LPCWSTR wszName,ULONG uFlags) {
    return This->lpVtbl->RemoveKey(This,wszName,uFlags);
}
static __WIDL_INLINE HRESULT IWbemPathKeyList_RemoveAllKeys(IWbemPathKeyList* This,ULONG uFlags) {
    return This->lpVtbl->RemoveAllKeys(This,uFlags);
}
static __WIDL_INLINE HRESULT IWbemPathKeyList_MakeSingleton(IWbemPathKeyList* This,boolean bSet) {
    return This->lpVtbl->MakeSingleton(This,bSet);
}
static __WIDL_INLINE HRESULT IWbemPathKeyList_GetInfo(IWbemPathKeyList* This,ULONG uRequestedInfo,ULONGLONG *puResponse) {
    return This->lpVtbl->GetInfo(This,uRequestedInfo,puResponse);
}
static __WIDL_INLINE HRESULT IWbemPathKeyList_GetText(IWbemPathKeyList* This,LONG lFlags,ULONG *puBuffLength,LPWSTR pszText) {
    return This->lpVtbl->GetText(This,lFlags,puBuffLength,pszText);
}
#endif
#endif

#endif


#endif  /* __IWbemPathKeyList_INTERFACE_DEFINED__ */

#ifdef WINE_NO_UNICODE_MACROS
#undef GetClassName
#endif
/*****************************************************************************
 * IWbemPath interface
 */
#ifndef __IWbemPath_INTERFACE_DEFINED__
#define __IWbemPath_INTERFACE_DEFINED__

DEFINE_GUID(IID_IWbemPath, 0x3bc15af2, 0x736c, 0x477e, 0x9e,0x51, 0x23,0x8a,0xf8,0x66,0x7d,0xcc);
#if defined(__cplusplus) && !defined(CINTERFACE)
MIDL_INTERFACE("3bc15af2-736c-477e-9e51-238af8667dcc")
IWbemPath : public IUnknown
{
    virtual HRESULT STDMETHODCALLTYPE SetText(
        ULONG uMode,
        LPCWSTR pszPath) = 0;

    virtual HRESULT STDMETHODCALLTYPE GetText(
        LONG lFlags,
        ULONG *puBuffLength,
        LPWSTR pszText) = 0;

    virtual HRESULT STDMETHODCALLTYPE GetInfo(
        ULONG uRequestedInfo,
        ULONGLONG *puResponse) = 0;

    virtual HRESULT STDMETHODCALLTYPE SetServer(
        LPCWSTR Name) = 0;

    virtual HRESULT STDMETHODCALLTYPE GetServer(
        ULONG *puNameBufLength,
        LPWSTR pName) = 0;

    virtual HRESULT STDMETHODCALLTYPE GetNamespaceCount(
        ULONG *puCount) = 0;

    virtual HRESULT STDMETHODCALLTYPE SetNamespaceAt(
        ULONG uIndex,
        LPCWSTR pszName) = 0;

    virtual HRESULT STDMETHODCALLTYPE GetNamespaceAt(
        ULONG uIndex,
        ULONG *puNameBufLength,
        LPWSTR pName) = 0;

    virtual HRESULT STDMETHODCALLTYPE RemoveNamespaceAt(
        ULONG uIndex) = 0;

    virtual HRESULT STDMETHODCALLTYPE RemoveAllNamespaces(
        ) = 0;

    virtual HRESULT STDMETHODCALLTYPE GetScopeCount(
        ULONG *puCount) = 0;

    virtual HRESULT STDMETHODCALLTYPE SetScope(
        ULONG uIndex,
        LPWSTR pszClass) = 0;

    virtual HRESULT STDMETHODCALLTYPE SetScopeFromText(
        ULONG uIndex,
        LPWSTR pszText) = 0;

    virtual HRESULT STDMETHODCALLTYPE GetScope(
        ULONG uIndex,
        ULONG *puClassNameBufSize,
        LPWSTR pszClass,
        IWbemPathKeyList **pKeyList) = 0;

    virtual HRESULT STDMETHODCALLTYPE GetScopeAsText(
        ULONG uIndex,
        ULONG *puTextBufSize,
        LPWSTR pszText) = 0;

    virtual HRESULT STDMETHODCALLTYPE RemoveScope(
        ULONG uIndex) = 0;

    virtual HRESULT STDMETHODCALLTYPE RemoveAllScopes(
        ) = 0;

    virtual HRESULT STDMETHODCALLTYPE SetClassName(
        LPCWSTR Name) = 0;

    virtual HRESULT STDMETHODCALLTYPE GetClassName(
        ULONG *puBuffLength,
        LPWSTR pszName) = 0;

    virtual HRESULT STDMETHODCALLTYPE GetKeyList(
        IWbemPathKeyList **pOut) = 0;

    virtual HRESULT STDMETHODCALLTYPE CreateClassPart(
        LONG lFlags,
        LPCWSTR Name) = 0;

    virtual HRESULT STDMETHODCALLTYPE DeleteClassPart(
        LONG lFlags) = 0;

    virtual BOOL STDMETHODCALLTYPE IsRelative(
        LPWSTR wszMachine,
        LPWSTR wszNamespace) = 0;

    virtual BOOL STDMETHODCALLTYPE IsRelativeOrChild(
        LPWSTR wszMachine,
        LPWSTR wszNamespace,
        LONG lFlags) = 0;

    virtual BOOL STDMETHODCALLTYPE IsLocal(
        LPCWSTR wszMachine) = 0;

    virtual BOOL STDMETHODCALLTYPE IsSameClassName(
        LPCWSTR wszClass) = 0;

};
#ifdef __CRT_UUID_DECL
__CRT_UUID_DECL(IWbemPath, 0x3bc15af2, 0x736c, 0x477e, 0x9e,0x51, 0x23,0x8a,0xf8,0x66,0x7d,0xcc)
#endif
#else
typedef struct IWbemPathVtbl {
    BEGIN_INTERFACE

    /*** IUnknown methods ***/
    HRESULT (STDMETHODCALLTYPE *QueryInterface)(
        IWbemPath *This,
        REFIID riid,
        void **ppvObject);

    ULONG (STDMETHODCALLTYPE *AddRef)(
        IWbemPath *This);

    ULONG (STDMETHODCALLTYPE *Release)(
        IWbemPath *This);

    /*** IWbemPath methods ***/
    HRESULT (STDMETHODCALLTYPE *SetText)(
        IWbemPath *This,
        ULONG uMode,
        LPCWSTR pszPath);

    HRESULT (STDMETHODCALLTYPE *GetText)(
        IWbemPath *This,
        LONG lFlags,
        ULONG *puBuffLength,
        LPWSTR pszText);

    HRESULT (STDMETHODCALLTYPE *GetInfo)(
        IWbemPath *This,
        ULONG uRequestedInfo,
        ULONGLONG *puResponse);

    HRESULT (STDMETHODCALLTYPE *SetServer)(
        IWbemPath *This,
        LPCWSTR Name);

    HRESULT (STDMETHODCALLTYPE *GetServer)(
        IWbemPath *This,
        ULONG *puNameBufLength,
        LPWSTR pName);

    HRESULT (STDMETHODCALLTYPE *GetNamespaceCount)(
        IWbemPath *This,
        ULONG *puCount);

    HRESULT (STDMETHODCALLTYPE *SetNamespaceAt)(
        IWbemPath *This,
        ULONG uIndex,
        LPCWSTR pszName);

    HRESULT (STDMETHODCALLTYPE *GetNamespaceAt)(
        IWbemPath *This,
        ULONG uIndex,
        ULONG *puNameBufLength,
        LPWSTR pName);

    HRESULT (STDMETHODCALLTYPE *RemoveNamespaceAt)(
        IWbemPath *This,
        ULONG uIndex);

    HRESULT (STDMETHODCALLTYPE *RemoveAllNamespaces)(
        IWbemPath *This);

    HRESULT (STDMETHODCALLTYPE *GetScopeCount)(
        IWbemPath *This,
        ULONG *puCount);

    HRESULT (STDMETHODCALLTYPE *SetScope)(
        IWbemPath *This,
        ULONG uIndex,
        LPWSTR pszClass);

    HRESULT (STDMETHODCALLTYPE *SetScopeFromText)(
        IWbemPath *This,
        ULONG uIndex,
        LPWSTR pszText);

    HRESULT (STDMETHODCALLTYPE *GetScope)(
        IWbemPath *This,
        ULONG uIndex,
        ULONG *puClassNameBufSize,
        LPWSTR pszClass,
        IWbemPathKeyList **pKeyList);

    HRESULT (STDMETHODCALLTYPE *GetScopeAsText)(
        IWbemPath *This,
        ULONG uIndex,
        ULONG *puTextBufSize,
        LPWSTR pszText);

    HRESULT (STDMETHODCALLTYPE *RemoveScope)(
        IWbemPath *This,
        ULONG uIndex);

    HRESULT (STDMETHODCALLTYPE *RemoveAllScopes)(
        IWbemPath *This);

    HRESULT (STDMETHODCALLTYPE *SetClassName)(
        IWbemPath *This,
        LPCWSTR Name);

    HRESULT (STDMETHODCALLTYPE *GetClassName)(
        IWbemPath *This,
        ULONG *puBuffLength,
        LPWSTR pszName);

    HRESULT (STDMETHODCALLTYPE *GetKeyList)(
        IWbemPath *This,
        IWbemPathKeyList **pOut);

    HRESULT (STDMETHODCALLTYPE *CreateClassPart)(
        IWbemPath *This,
        LONG lFlags,
        LPCWSTR Name);

    HRESULT (STDMETHODCALLTYPE *DeleteClassPart)(
        IWbemPath *This,
        LONG lFlags);

    BOOL (STDMETHODCALLTYPE *IsRelative)(
        IWbemPath *This,
        LPWSTR wszMachine,
        LPWSTR wszNamespace);

    BOOL (STDMETHODCALLTYPE *IsRelativeOrChild)(
        IWbemPath *This,
        LPWSTR wszMachine,
        LPWSTR wszNamespace,
        LONG lFlags);

    BOOL (STDMETHODCALLTYPE *IsLocal)(
        IWbemPath *This,
        LPCWSTR wszMachine);

    BOOL (STDMETHODCALLTYPE *IsSameClassName)(
        IWbemPath *This,
        LPCWSTR wszClass);

    END_INTERFACE
} IWbemPathVtbl;

interface IWbemPath {
    CONST_VTBL IWbemPathVtbl* lpVtbl;
};

#ifdef COBJMACROS
#ifndef WIDL_C_INLINE_WRAPPERS
/*** IUnknown methods ***/
#define IWbemPath_QueryInterface(This,riid,ppvObject) (This)->lpVtbl->QueryInterface(This,riid,ppvObject)
#define IWbemPath_AddRef(This) (This)->lpVtbl->AddRef(This)
#define IWbemPath_Release(This) (This)->lpVtbl->Release(This)
/*** IWbemPath methods ***/
#define IWbemPath_SetText(This,uMode,pszPath) (This)->lpVtbl->SetText(This,uMode,pszPath)
#define IWbemPath_GetText(This,lFlags,puBuffLength,pszText) (This)->lpVtbl->GetText(This,lFlags,puBuffLength,pszText)
#define IWbemPath_GetInfo(This,uRequestedInfo,puResponse) (This)->lpVtbl->GetInfo(This,uRequestedInfo,puResponse)
#define IWbemPath_SetServer(This,Name) (This)->lpVtbl->SetServer(This,Name)
#define IWbemPath_GetServer(This,puNameBufLength,pName) (This)->lpVtbl->GetServer(This,puNameBufLength,pName)
#define IWbemPath_GetNamespaceCount(This,puCount) (This)->lpVtbl->GetNamespaceCount(This,puCount)
#define IWbemPath_SetNamespaceAt(This,uIndex,pszName) (This)->lpVtbl->SetNamespaceAt(This,uIndex,pszName)
#define IWbemPath_GetNamespaceAt(This,uIndex,puNameBufLength,pName) (This)->lpVtbl->GetNamespaceAt(This,uIndex,puNameBufLength,pName)
#define IWbemPath_RemoveNamespaceAt(This,uIndex) (This)->lpVtbl->RemoveNamespaceAt(This,uIndex)
#define IWbemPath_RemoveAllNamespaces(This) (This)->lpVtbl->RemoveAllNamespaces(This)
#define IWbemPath_GetScopeCount(This,puCount) (This)->lpVtbl->GetScopeCount(This,puCount)
#define IWbemPath_SetScope(This,uIndex,pszClass) (This)->lpVtbl->SetScope(This,uIndex,pszClass)
#define IWbemPath_SetScopeFromText(This,uIndex,pszText) (This)->lpVtbl->SetScopeFromText(This,uIndex,pszText)
#define IWbemPath_GetScope(This,uIndex,puClassNameBufSize,pszClass,pKeyList) (This)->lpVtbl->GetScope(This,uIndex,puClassNameBufSize,pszClass,pKeyList)
#define IWbemPath_GetScopeAsText(This,uIndex,puTextBufSize,pszText) (This)->lpVtbl->GetScopeAsText(This,uIndex,puTextBufSize,pszText)
#define IWbemPath_RemoveScope(This,uIndex) (This)->lpVtbl->RemoveScope(This,uIndex)
#define IWbemPath_RemoveAllScopes(This) (This)->lpVtbl->RemoveAllScopes(This)
#define IWbemPath_SetClassName(This,Name) (This)->lpVtbl->SetClassName(This,Name)
#define IWbemPath_GetClassName(This,puBuffLength,pszName) (This)->lpVtbl->GetClassName(This,puBuffLength,pszName)
#define IWbemPath_GetKeyList(This,pOut) (This)->lpVtbl->GetKeyList(This,pOut)
#define IWbemPath_CreateClassPart(This,lFlags,Name) (This)->lpVtbl->CreateClassPart(This,lFlags,Name)
#define IWbemPath_DeleteClassPart(This,lFlags) (This)->lpVtbl->DeleteClassPart(This,lFlags)
#define IWbemPath_IsRelative(This,wszMachine,wszNamespace) (This)->lpVtbl->IsRelative(This,wszMachine,wszNamespace)
#define IWbemPath_IsRelativeOrChild(This,wszMachine,wszNamespace,lFlags) (This)->lpVtbl->IsRelativeOrChild(This,wszMachine,wszNamespace,lFlags)
#define IWbemPath_IsLocal(This,wszMachine) (This)->lpVtbl->IsLocal(This,wszMachine)
#define IWbemPath_IsSameClassName(This,wszClass) (This)->lpVtbl->IsSameClassName(This,wszClass)
#else
/*** IUnknown methods ***/
static __WIDL_INLINE HRESULT IWbemPath_QueryInterface(IWbemPath* This,REFIID riid,void **ppvObject) {
    return This->lpVtbl->QueryInterface(This,riid,ppvObject);
}
static __WIDL_INLINE ULONG IWbemPath_AddRef(IWbemPath* This) {
    return This->lpVtbl->AddRef(This);
}
static __WIDL_INLINE ULONG IWbemPath_Release(IWbemPath* This) {
    return This->lpVtbl->Release(This);
}
/*** IWbemPath methods ***/
static __WIDL_INLINE HRESULT IWbemPath_SetText(IWbemPath* This,ULONG uMode,LPCWSTR pszPath) {
    return This->lpVtbl->SetText(This,uMode,pszPath);
}
static __WIDL_INLINE HRESULT IWbemPath_GetText(IWbemPath* This,LONG lFlags,ULONG *puBuffLength,LPWSTR pszText) {
    return This->lpVtbl->GetText(This,lFlags,puBuffLength,pszText);
}
static __WIDL_INLINE HRESULT IWbemPath_GetInfo(IWbemPath* This,ULONG uRequestedInfo,ULONGLONG *puResponse) {
    return This->lpVtbl->GetInfo(This,uRequestedInfo,puResponse);
}
static __WIDL_INLINE HRESULT IWbemPath_SetServer(IWbemPath* This,LPCWSTR Name) {
    return This->lpVtbl->SetServer(This,Name);
}
static __WIDL_INLINE HRESULT IWbemPath_GetServer(IWbemPath* This,ULONG *puNameBufLength,LPWSTR pName) {
    return This->lpVtbl->GetServer(This,puNameBufLength,pName);
}
static __WIDL_INLINE HRESULT IWbemPath_GetNamespaceCount(IWbemPath* This,ULONG *puCount) {
    return This->lpVtbl->GetNamespaceCount(This,puCount);
}
static __WIDL_INLINE HRESULT IWbemPath_SetNamespaceAt(IWbemPath* This,ULONG uIndex,LPCWSTR pszName) {
    return This->lpVtbl->SetNamespaceAt(This,uIndex,pszName);
}
static __WIDL_INLINE HRESULT IWbemPath_GetNamespaceAt(IWbemPath* This,ULONG uIndex,ULONG *puNameBufLength,LPWSTR pName) {
    return This->lpVtbl->GetNamespaceAt(This,uIndex,puNameBufLength,pName);
}
static __WIDL_INLINE HRESULT IWbemPath_RemoveNamespaceAt(IWbemPath* This,ULONG uIndex) {
    return This->lpVtbl->RemoveNamespaceAt(This,uIndex);
}
static __WIDL_INLINE HRESULT IWbemPath_RemoveAllNamespaces(IWbemPath* This) {
    return This->lpVtbl->RemoveAllNamespaces(This);
}
static __WIDL_INLINE HRESULT IWbemPath_GetScopeCount(IWbemPath* This,ULONG *puCount) {
    return This->lpVtbl->GetScopeCount(This,puCount);
}
static __WIDL_INLINE HRESULT IWbemPath_SetScope(IWbemPath* This,ULONG uIndex,LPWSTR pszClass) {
    return This->lpVtbl->SetScope(This,uIndex,pszClass);
}
static __WIDL_INLINE HRESULT IWbemPath_SetScopeFromText(IWbemPath* This,ULONG uIndex,LPWSTR pszText) {
    return This->lpVtbl->SetScopeFromText(This,uIndex,pszText);
}
static __WIDL_INLINE HRESULT IWbemPath_GetScope(IWbemPath* This,ULONG uIndex,ULONG *puClassNameBufSize,LPWSTR pszClass,IWbemPathKeyList **pKeyList) {
    return This->lpVtbl->GetScope(This,uIndex,puClassNameBufSize,pszClass,pKeyList);
}
static __WIDL_INLINE HRESULT IWbemPath_GetScopeAsText(IWbemPath* This,ULONG uIndex,ULONG *puTextBufSize,LPWSTR pszText) {
    return This->lpVtbl->GetScopeAsText(This,uIndex,puTextBufSize,pszText);
}
static __WIDL_INLINE HRESULT IWbemPath_RemoveScope(IWbemPath* This,ULONG uIndex) {
    return This->lpVtbl->RemoveScope(This,uIndex);
}
static __WIDL_INLINE HRESULT IWbemPath_RemoveAllScopes(IWbemPath* This) {
    return This->lpVtbl->RemoveAllScopes(This);
}
static __WIDL_INLINE HRESULT IWbemPath_SetClassName(IWbemPath* This,LPCWSTR Name) {
    return This->lpVtbl->SetClassName(This,Name);
}
static __WIDL_INLINE HRESULT IWbemPath_GetClassName(IWbemPath* This,ULONG *puBuffLength,LPWSTR pszName) {
    return This->lpVtbl->GetClassName(This,puBuffLength,pszName);
}
static __WIDL_INLINE HRESULT IWbemPath_GetKeyList(IWbemPath* This,IWbemPathKeyList **pOut) {
    return This->lpVtbl->GetKeyList(This,pOut);
}
static __WIDL_INLINE HRESULT IWbemPath_CreateClassPart(IWbemPath* This,LONG lFlags,LPCWSTR Name) {
    return This->lpVtbl->CreateClassPart(This,lFlags,Name);
}
static __WIDL_INLINE HRESULT IWbemPath_DeleteClassPart(IWbemPath* This,LONG lFlags) {
    return This->lpVtbl->DeleteClassPart(This,lFlags);
}
static __WIDL_INLINE BOOL IWbemPath_IsRelative(IWbemPath* This,LPWSTR wszMachine,LPWSTR wszNamespace) {
    return This->lpVtbl->IsRelative(This,wszMachine,wszNamespace);
}
static __WIDL_INLINE BOOL IWbemPath_IsRelativeOrChild(IWbemPath* This,LPWSTR wszMachine,LPWSTR wszNamespace,LONG lFlags) {
    return This->lpVtbl->IsRelativeOrChild(This,wszMachine,wszNamespace,lFlags);
}
static __WIDL_INLINE BOOL IWbemPath_IsLocal(IWbemPath* This,LPCWSTR wszMachine) {
    return This->lpVtbl->IsLocal(This,wszMachine);
}
static __WIDL_INLINE BOOL IWbemPath_IsSameClassName(IWbemPath* This,LPCWSTR wszClass) {
    return This->lpVtbl->IsSameClassName(This,wszClass);
}
#endif
#endif

#endif


#endif  /* __IWbemPath_INTERFACE_DEFINED__ */

/*****************************************************************************
 * WbemDefPath coclass
 */

DEFINE_GUID(CLSID_WbemDefPath, 0xcf4cc405, 0xe2c5, 0x4ddd, 0xb3,0xce, 0x5e,0x75,0x82,0xd8,0xc9,0xfa);

#ifdef __cplusplus
class DECLSPEC_UUID("cf4cc405-e2c5-4ddd-b3ce-5e7582d8c9fa") WbemDefPath;
#ifdef __CRT_UUID_DECL
__CRT_UUID_DECL(WbemDefPath, 0xcf4cc405, 0xe2c5, 0x4ddd, 0xb3,0xce, 0x5e,0x75,0x82,0xd8,0xc9,0xfa)
#endif
#endif

/* Begin additional prototypes for all interfaces */


/* End additional prototypes */

#ifdef __cplusplus
}
#endif

#endif /* __wmiutils_h__ */
