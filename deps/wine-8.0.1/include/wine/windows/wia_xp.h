/*** Autogenerated by WIDL 8.0.1 from ../include/wia_xp.idl - Do not edit ***/

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

#ifndef __wia_xp_h__
#define __wia_xp_h__

#ifndef __WIDL_INLINE
#if defined(__cplusplus) || defined(_MSC_VER)
#define __WIDL_INLINE inline
#elif defined(__GNUC__)
#define __WIDL_INLINE __inline__
#endif
#endif

/* Forward declarations */

#ifndef __IWiaDevMgr_FWD_DEFINED__
#define __IWiaDevMgr_FWD_DEFINED__
typedef interface IWiaDevMgr IWiaDevMgr;
#ifdef __cplusplus
interface IWiaDevMgr;
#endif /* __cplusplus */
#endif

#ifndef __IEnumWIA_DEV_INFO_FWD_DEFINED__
#define __IEnumWIA_DEV_INFO_FWD_DEFINED__
typedef interface IEnumWIA_DEV_INFO IEnumWIA_DEV_INFO;
#ifdef __cplusplus
interface IEnumWIA_DEV_INFO;
#endif /* __cplusplus */
#endif

#ifndef __IWiaPropertyStorage_FWD_DEFINED__
#define __IWiaPropertyStorage_FWD_DEFINED__
typedef interface IWiaPropertyStorage IWiaPropertyStorage;
#ifdef __cplusplus
interface IWiaPropertyStorage;
#endif /* __cplusplus */
#endif

#ifndef __IWiaItem_FWD_DEFINED__
#define __IWiaItem_FWD_DEFINED__
typedef interface IWiaItem IWiaItem;
#ifdef __cplusplus
interface IWiaItem;
#endif /* __cplusplus */
#endif

#ifndef __IWiaEventCallback_FWD_DEFINED__
#define __IWiaEventCallback_FWD_DEFINED__
typedef interface IWiaEventCallback IWiaEventCallback;
#ifdef __cplusplus
interface IWiaEventCallback;
#endif /* __cplusplus */
#endif

/* Headers for imported files */

#include <unknwn.h>
#include <oaidl.h>
#include <propidl.h>

#ifdef __cplusplus
extern "C" {
#endif

#include <wiadef.h>
#ifndef __IEnumWIA_DEV_INFO_FWD_DEFINED__
#define __IEnumWIA_DEV_INFO_FWD_DEFINED__
typedef interface IEnumWIA_DEV_INFO IEnumWIA_DEV_INFO;
#ifdef __cplusplus
interface IEnumWIA_DEV_INFO;
#endif /* __cplusplus */
#endif

#ifndef __IWiaPropertyStorage_FWD_DEFINED__
#define __IWiaPropertyStorage_FWD_DEFINED__
typedef interface IWiaPropertyStorage IWiaPropertyStorage;
#ifdef __cplusplus
interface IWiaPropertyStorage;
#endif /* __cplusplus */
#endif

#ifndef __IWiaItem_FWD_DEFINED__
#define __IWiaItem_FWD_DEFINED__
typedef interface IWiaItem IWiaItem;
#ifdef __cplusplus
interface IWiaItem;
#endif /* __cplusplus */
#endif

#ifndef __IWiaEventCallback_FWD_DEFINED__
#define __IWiaEventCallback_FWD_DEFINED__
typedef interface IWiaEventCallback IWiaEventCallback;
#ifdef __cplusplus
interface IWiaEventCallback;
#endif /* __cplusplus */
#endif

DEFINE_GUID(CLSID_WiaDevMgr, 0xa1f4e726,0x8cf1,0x11d1,0xbf,0x92,0x00,0x60,0x08,0x1e,0xd8,0x11);
/*****************************************************************************
 * IWiaDevMgr interface
 */
#ifndef __IWiaDevMgr_INTERFACE_DEFINED__
#define __IWiaDevMgr_INTERFACE_DEFINED__

DEFINE_GUID(IID_IWiaDevMgr, 0x5eb2502a, 0x8cf1, 0x11d1, 0xbf,0x92, 0x00,0x60,0x08,0x1e,0xd8,0x11);
#if defined(__cplusplus) && !defined(CINTERFACE)
MIDL_INTERFACE("5eb2502a-8cf1-11d1-bf92-0060081ed811")
IWiaDevMgr : public IUnknown
{
    virtual HRESULT STDMETHODCALLTYPE EnumDeviceInfo(
        LONG lFlag,
        IEnumWIA_DEV_INFO **ppIEnum) = 0;

    virtual HRESULT STDMETHODCALLTYPE CreateDevice(
        BSTR bstrDeviceID,
        IWiaItem **ppWiaItemRoot) = 0;

    virtual HRESULT STDMETHODCALLTYPE SelectDeviceDlg(
        HWND hwndParent,
        LONG lDeviceType,
        LONG lFlags,
        BSTR *pbstrDeviceID,
        IWiaItem **ppItemRoot) = 0;

    virtual HRESULT STDMETHODCALLTYPE SelectDeviceDlgID(
        HWND hwndParent,
        LONG lDeviceType,
        LONG lFlags,
        BSTR *pbstrDeviceID) = 0;

    virtual HRESULT STDMETHODCALLTYPE GetImageDlg(
        HWND hwndParent,
        LONG lDeviceType,
        LONG lFlags,
        LONG lIntent,
        IWiaItem *pItemRoot,
        BSTR bstrFilename,
        GUID *pguidFormat) = 0;

    virtual HRESULT STDMETHODCALLTYPE RegisterEventCallbackProgram(
        LONG lFlags,
        BSTR bstrDeviceID,
        const GUID *pEventGUID,
        BSTR bstrCommandline,
        BSTR bstrName,
        BSTR bstrDescription,
        BSTR bstrIcon) = 0;

    virtual HRESULT STDMETHODCALLTYPE RegisterEventCallbackInterface(
        LONG lFlags,
        BSTR bstrDeviceID,
        const GUID *pEventGUID,
        IWiaEventCallback *pIWiaEventCallback,
        IUnknown **pEventObject) = 0;

    virtual HRESULT STDMETHODCALLTYPE RegisterEventCallbackCLSID(
        LONG lFlags,
        BSTR bstrDeviceID,
        const GUID *pEventGUID,
        const GUID *pClsID,
        BSTR bstrName,
        BSTR bstrDescription,
        BSTR bstrIcon) = 0;

    virtual HRESULT STDMETHODCALLTYPE AddDeviceDlg(
        HWND hwndParent,
        LONG lFlags) = 0;

};
#ifdef __CRT_UUID_DECL
__CRT_UUID_DECL(IWiaDevMgr, 0x5eb2502a, 0x8cf1, 0x11d1, 0xbf,0x92, 0x00,0x60,0x08,0x1e,0xd8,0x11)
#endif
#else
typedef struct IWiaDevMgrVtbl {
    BEGIN_INTERFACE

    /*** IUnknown methods ***/
    HRESULT (STDMETHODCALLTYPE *QueryInterface)(
        IWiaDevMgr *This,
        REFIID riid,
        void **ppvObject);

    ULONG (STDMETHODCALLTYPE *AddRef)(
        IWiaDevMgr *This);

    ULONG (STDMETHODCALLTYPE *Release)(
        IWiaDevMgr *This);

    /*** IWiaDevMgr methods ***/
    HRESULT (STDMETHODCALLTYPE *EnumDeviceInfo)(
        IWiaDevMgr *This,
        LONG lFlag,
        IEnumWIA_DEV_INFO **ppIEnum);

    HRESULT (STDMETHODCALLTYPE *CreateDevice)(
        IWiaDevMgr *This,
        BSTR bstrDeviceID,
        IWiaItem **ppWiaItemRoot);

    HRESULT (STDMETHODCALLTYPE *SelectDeviceDlg)(
        IWiaDevMgr *This,
        HWND hwndParent,
        LONG lDeviceType,
        LONG lFlags,
        BSTR *pbstrDeviceID,
        IWiaItem **ppItemRoot);

    HRESULT (STDMETHODCALLTYPE *SelectDeviceDlgID)(
        IWiaDevMgr *This,
        HWND hwndParent,
        LONG lDeviceType,
        LONG lFlags,
        BSTR *pbstrDeviceID);

    HRESULT (STDMETHODCALLTYPE *GetImageDlg)(
        IWiaDevMgr *This,
        HWND hwndParent,
        LONG lDeviceType,
        LONG lFlags,
        LONG lIntent,
        IWiaItem *pItemRoot,
        BSTR bstrFilename,
        GUID *pguidFormat);

    HRESULT (STDMETHODCALLTYPE *RegisterEventCallbackProgram)(
        IWiaDevMgr *This,
        LONG lFlags,
        BSTR bstrDeviceID,
        const GUID *pEventGUID,
        BSTR bstrCommandline,
        BSTR bstrName,
        BSTR bstrDescription,
        BSTR bstrIcon);

    HRESULT (STDMETHODCALLTYPE *RegisterEventCallbackInterface)(
        IWiaDevMgr *This,
        LONG lFlags,
        BSTR bstrDeviceID,
        const GUID *pEventGUID,
        IWiaEventCallback *pIWiaEventCallback,
        IUnknown **pEventObject);

    HRESULT (STDMETHODCALLTYPE *RegisterEventCallbackCLSID)(
        IWiaDevMgr *This,
        LONG lFlags,
        BSTR bstrDeviceID,
        const GUID *pEventGUID,
        const GUID *pClsID,
        BSTR bstrName,
        BSTR bstrDescription,
        BSTR bstrIcon);

    HRESULT (STDMETHODCALLTYPE *AddDeviceDlg)(
        IWiaDevMgr *This,
        HWND hwndParent,
        LONG lFlags);

    END_INTERFACE
} IWiaDevMgrVtbl;

interface IWiaDevMgr {
    CONST_VTBL IWiaDevMgrVtbl* lpVtbl;
};

#ifdef COBJMACROS
#ifndef WIDL_C_INLINE_WRAPPERS
/*** IUnknown methods ***/
#define IWiaDevMgr_QueryInterface(This,riid,ppvObject) (This)->lpVtbl->QueryInterface(This,riid,ppvObject)
#define IWiaDevMgr_AddRef(This) (This)->lpVtbl->AddRef(This)
#define IWiaDevMgr_Release(This) (This)->lpVtbl->Release(This)
/*** IWiaDevMgr methods ***/
#define IWiaDevMgr_EnumDeviceInfo(This,lFlag,ppIEnum) (This)->lpVtbl->EnumDeviceInfo(This,lFlag,ppIEnum)
#define IWiaDevMgr_CreateDevice(This,bstrDeviceID,ppWiaItemRoot) (This)->lpVtbl->CreateDevice(This,bstrDeviceID,ppWiaItemRoot)
#define IWiaDevMgr_SelectDeviceDlg(This,hwndParent,lDeviceType,lFlags,pbstrDeviceID,ppItemRoot) (This)->lpVtbl->SelectDeviceDlg(This,hwndParent,lDeviceType,lFlags,pbstrDeviceID,ppItemRoot)
#define IWiaDevMgr_SelectDeviceDlgID(This,hwndParent,lDeviceType,lFlags,pbstrDeviceID) (This)->lpVtbl->SelectDeviceDlgID(This,hwndParent,lDeviceType,lFlags,pbstrDeviceID)
#define IWiaDevMgr_GetImageDlg(This,hwndParent,lDeviceType,lFlags,lIntent,pItemRoot,bstrFilename,pguidFormat) (This)->lpVtbl->GetImageDlg(This,hwndParent,lDeviceType,lFlags,lIntent,pItemRoot,bstrFilename,pguidFormat)
#define IWiaDevMgr_RegisterEventCallbackProgram(This,lFlags,bstrDeviceID,pEventGUID,bstrCommandline,bstrName,bstrDescription,bstrIcon) (This)->lpVtbl->RegisterEventCallbackProgram(This,lFlags,bstrDeviceID,pEventGUID,bstrCommandline,bstrName,bstrDescription,bstrIcon)
#define IWiaDevMgr_RegisterEventCallbackInterface(This,lFlags,bstrDeviceID,pEventGUID,pIWiaEventCallback,pEventObject) (This)->lpVtbl->RegisterEventCallbackInterface(This,lFlags,bstrDeviceID,pEventGUID,pIWiaEventCallback,pEventObject)
#define IWiaDevMgr_RegisterEventCallbackCLSID(This,lFlags,bstrDeviceID,pEventGUID,pClsID,bstrName,bstrDescription,bstrIcon) (This)->lpVtbl->RegisterEventCallbackCLSID(This,lFlags,bstrDeviceID,pEventGUID,pClsID,bstrName,bstrDescription,bstrIcon)
#define IWiaDevMgr_AddDeviceDlg(This,hwndParent,lFlags) (This)->lpVtbl->AddDeviceDlg(This,hwndParent,lFlags)
#else
/*** IUnknown methods ***/
static __WIDL_INLINE HRESULT IWiaDevMgr_QueryInterface(IWiaDevMgr* This,REFIID riid,void **ppvObject) {
    return This->lpVtbl->QueryInterface(This,riid,ppvObject);
}
static __WIDL_INLINE ULONG IWiaDevMgr_AddRef(IWiaDevMgr* This) {
    return This->lpVtbl->AddRef(This);
}
static __WIDL_INLINE ULONG IWiaDevMgr_Release(IWiaDevMgr* This) {
    return This->lpVtbl->Release(This);
}
/*** IWiaDevMgr methods ***/
static __WIDL_INLINE HRESULT IWiaDevMgr_EnumDeviceInfo(IWiaDevMgr* This,LONG lFlag,IEnumWIA_DEV_INFO **ppIEnum) {
    return This->lpVtbl->EnumDeviceInfo(This,lFlag,ppIEnum);
}
static __WIDL_INLINE HRESULT IWiaDevMgr_CreateDevice(IWiaDevMgr* This,BSTR bstrDeviceID,IWiaItem **ppWiaItemRoot) {
    return This->lpVtbl->CreateDevice(This,bstrDeviceID,ppWiaItemRoot);
}
static __WIDL_INLINE HRESULT IWiaDevMgr_SelectDeviceDlg(IWiaDevMgr* This,HWND hwndParent,LONG lDeviceType,LONG lFlags,BSTR *pbstrDeviceID,IWiaItem **ppItemRoot) {
    return This->lpVtbl->SelectDeviceDlg(This,hwndParent,lDeviceType,lFlags,pbstrDeviceID,ppItemRoot);
}
static __WIDL_INLINE HRESULT IWiaDevMgr_SelectDeviceDlgID(IWiaDevMgr* This,HWND hwndParent,LONG lDeviceType,LONG lFlags,BSTR *pbstrDeviceID) {
    return This->lpVtbl->SelectDeviceDlgID(This,hwndParent,lDeviceType,lFlags,pbstrDeviceID);
}
static __WIDL_INLINE HRESULT IWiaDevMgr_GetImageDlg(IWiaDevMgr* This,HWND hwndParent,LONG lDeviceType,LONG lFlags,LONG lIntent,IWiaItem *pItemRoot,BSTR bstrFilename,GUID *pguidFormat) {
    return This->lpVtbl->GetImageDlg(This,hwndParent,lDeviceType,lFlags,lIntent,pItemRoot,bstrFilename,pguidFormat);
}
static __WIDL_INLINE HRESULT IWiaDevMgr_RegisterEventCallbackProgram(IWiaDevMgr* This,LONG lFlags,BSTR bstrDeviceID,const GUID *pEventGUID,BSTR bstrCommandline,BSTR bstrName,BSTR bstrDescription,BSTR bstrIcon) {
    return This->lpVtbl->RegisterEventCallbackProgram(This,lFlags,bstrDeviceID,pEventGUID,bstrCommandline,bstrName,bstrDescription,bstrIcon);
}
static __WIDL_INLINE HRESULT IWiaDevMgr_RegisterEventCallbackInterface(IWiaDevMgr* This,LONG lFlags,BSTR bstrDeviceID,const GUID *pEventGUID,IWiaEventCallback *pIWiaEventCallback,IUnknown **pEventObject) {
    return This->lpVtbl->RegisterEventCallbackInterface(This,lFlags,bstrDeviceID,pEventGUID,pIWiaEventCallback,pEventObject);
}
static __WIDL_INLINE HRESULT IWiaDevMgr_RegisterEventCallbackCLSID(IWiaDevMgr* This,LONG lFlags,BSTR bstrDeviceID,const GUID *pEventGUID,const GUID *pClsID,BSTR bstrName,BSTR bstrDescription,BSTR bstrIcon) {
    return This->lpVtbl->RegisterEventCallbackCLSID(This,lFlags,bstrDeviceID,pEventGUID,pClsID,bstrName,bstrDescription,bstrIcon);
}
static __WIDL_INLINE HRESULT IWiaDevMgr_AddDeviceDlg(IWiaDevMgr* This,HWND hwndParent,LONG lFlags) {
    return This->lpVtbl->AddDeviceDlg(This,hwndParent,lFlags);
}
#endif
#endif

#endif


#endif  /* __IWiaDevMgr_INTERFACE_DEFINED__ */

/*****************************************************************************
 * IEnumWIA_DEV_INFO interface
 */
#ifndef __IEnumWIA_DEV_INFO_INTERFACE_DEFINED__
#define __IEnumWIA_DEV_INFO_INTERFACE_DEFINED__

DEFINE_GUID(IID_IEnumWIA_DEV_INFO, 0x5e38b83c, 0x8cf1, 0x11d1, 0xbf,0x92, 0x00,0x60,0x08,0x1e,0xd8,0x11);
#if defined(__cplusplus) && !defined(CINTERFACE)
MIDL_INTERFACE("5e38b83c-8cf1-11d1-bf92-0060081ed811")
IEnumWIA_DEV_INFO : public IUnknown
{
    virtual HRESULT STDMETHODCALLTYPE Next(
        ULONG celt,
        IWiaPropertyStorage **rgelt,
        ULONG *pceltFetched) = 0;

    virtual HRESULT STDMETHODCALLTYPE Skip(
        ULONG celt) = 0;

    virtual HRESULT STDMETHODCALLTYPE Reset(
        ) = 0;

    virtual HRESULT STDMETHODCALLTYPE Clone(
        IEnumWIA_DEV_INFO **ppIEnum) = 0;

    virtual HRESULT STDMETHODCALLTYPE GetCount(
        ULONG *celt) = 0;

};
#ifdef __CRT_UUID_DECL
__CRT_UUID_DECL(IEnumWIA_DEV_INFO, 0x5e38b83c, 0x8cf1, 0x11d1, 0xbf,0x92, 0x00,0x60,0x08,0x1e,0xd8,0x11)
#endif
#else
typedef struct IEnumWIA_DEV_INFOVtbl {
    BEGIN_INTERFACE

    /*** IUnknown methods ***/
    HRESULT (STDMETHODCALLTYPE *QueryInterface)(
        IEnumWIA_DEV_INFO *This,
        REFIID riid,
        void **ppvObject);

    ULONG (STDMETHODCALLTYPE *AddRef)(
        IEnumWIA_DEV_INFO *This);

    ULONG (STDMETHODCALLTYPE *Release)(
        IEnumWIA_DEV_INFO *This);

    /*** IEnumWIA_DEV_INFO methods ***/
    HRESULT (STDMETHODCALLTYPE *Next)(
        IEnumWIA_DEV_INFO *This,
        ULONG celt,
        IWiaPropertyStorage **rgelt,
        ULONG *pceltFetched);

    HRESULT (STDMETHODCALLTYPE *Skip)(
        IEnumWIA_DEV_INFO *This,
        ULONG celt);

    HRESULT (STDMETHODCALLTYPE *Reset)(
        IEnumWIA_DEV_INFO *This);

    HRESULT (STDMETHODCALLTYPE *Clone)(
        IEnumWIA_DEV_INFO *This,
        IEnumWIA_DEV_INFO **ppIEnum);

    HRESULT (STDMETHODCALLTYPE *GetCount)(
        IEnumWIA_DEV_INFO *This,
        ULONG *celt);

    END_INTERFACE
} IEnumWIA_DEV_INFOVtbl;

interface IEnumWIA_DEV_INFO {
    CONST_VTBL IEnumWIA_DEV_INFOVtbl* lpVtbl;
};

#ifdef COBJMACROS
#ifndef WIDL_C_INLINE_WRAPPERS
/*** IUnknown methods ***/
#define IEnumWIA_DEV_INFO_QueryInterface(This,riid,ppvObject) (This)->lpVtbl->QueryInterface(This,riid,ppvObject)
#define IEnumWIA_DEV_INFO_AddRef(This) (This)->lpVtbl->AddRef(This)
#define IEnumWIA_DEV_INFO_Release(This) (This)->lpVtbl->Release(This)
/*** IEnumWIA_DEV_INFO methods ***/
#define IEnumWIA_DEV_INFO_Next(This,celt,rgelt,pceltFetched) (This)->lpVtbl->Next(This,celt,rgelt,pceltFetched)
#define IEnumWIA_DEV_INFO_Skip(This,celt) (This)->lpVtbl->Skip(This,celt)
#define IEnumWIA_DEV_INFO_Reset(This) (This)->lpVtbl->Reset(This)
#define IEnumWIA_DEV_INFO_Clone(This,ppIEnum) (This)->lpVtbl->Clone(This,ppIEnum)
#define IEnumWIA_DEV_INFO_GetCount(This,celt) (This)->lpVtbl->GetCount(This,celt)
#else
/*** IUnknown methods ***/
static __WIDL_INLINE HRESULT IEnumWIA_DEV_INFO_QueryInterface(IEnumWIA_DEV_INFO* This,REFIID riid,void **ppvObject) {
    return This->lpVtbl->QueryInterface(This,riid,ppvObject);
}
static __WIDL_INLINE ULONG IEnumWIA_DEV_INFO_AddRef(IEnumWIA_DEV_INFO* This) {
    return This->lpVtbl->AddRef(This);
}
static __WIDL_INLINE ULONG IEnumWIA_DEV_INFO_Release(IEnumWIA_DEV_INFO* This) {
    return This->lpVtbl->Release(This);
}
/*** IEnumWIA_DEV_INFO methods ***/
static __WIDL_INLINE HRESULT IEnumWIA_DEV_INFO_Next(IEnumWIA_DEV_INFO* This,ULONG celt,IWiaPropertyStorage **rgelt,ULONG *pceltFetched) {
    return This->lpVtbl->Next(This,celt,rgelt,pceltFetched);
}
static __WIDL_INLINE HRESULT IEnumWIA_DEV_INFO_Skip(IEnumWIA_DEV_INFO* This,ULONG celt) {
    return This->lpVtbl->Skip(This,celt);
}
static __WIDL_INLINE HRESULT IEnumWIA_DEV_INFO_Reset(IEnumWIA_DEV_INFO* This) {
    return This->lpVtbl->Reset(This);
}
static __WIDL_INLINE HRESULT IEnumWIA_DEV_INFO_Clone(IEnumWIA_DEV_INFO* This,IEnumWIA_DEV_INFO **ppIEnum) {
    return This->lpVtbl->Clone(This,ppIEnum);
}
static __WIDL_INLINE HRESULT IEnumWIA_DEV_INFO_GetCount(IEnumWIA_DEV_INFO* This,ULONG *celt) {
    return This->lpVtbl->GetCount(This,celt);
}
#endif
#endif

#endif


#endif  /* __IEnumWIA_DEV_INFO_INTERFACE_DEFINED__ */

/*****************************************************************************
 * IWiaPropertyStorage interface
 */
#ifndef __IWiaPropertyStorage_INTERFACE_DEFINED__
#define __IWiaPropertyStorage_INTERFACE_DEFINED__

DEFINE_GUID(IID_IWiaPropertyStorage, 0x98b5e8a0, 0x29cc, 0x491a, 0xaa,0xc0, 0xe6,0xdb,0x4f,0xdc,0xce,0xb6);
#if defined(__cplusplus) && !defined(CINTERFACE)
MIDL_INTERFACE("98b5e8a0-29cc-491a-aac0-e6db4fdcceb6")
IWiaPropertyStorage : public IUnknown
{
};
#ifdef __CRT_UUID_DECL
__CRT_UUID_DECL(IWiaPropertyStorage, 0x98b5e8a0, 0x29cc, 0x491a, 0xaa,0xc0, 0xe6,0xdb,0x4f,0xdc,0xce,0xb6)
#endif
#else
typedef struct IWiaPropertyStorageVtbl {
    BEGIN_INTERFACE

    /*** IUnknown methods ***/
    HRESULT (STDMETHODCALLTYPE *QueryInterface)(
        IWiaPropertyStorage *This,
        REFIID riid,
        void **ppvObject);

    ULONG (STDMETHODCALLTYPE *AddRef)(
        IWiaPropertyStorage *This);

    ULONG (STDMETHODCALLTYPE *Release)(
        IWiaPropertyStorage *This);

    END_INTERFACE
} IWiaPropertyStorageVtbl;

interface IWiaPropertyStorage {
    CONST_VTBL IWiaPropertyStorageVtbl* lpVtbl;
};

#ifdef COBJMACROS
#ifndef WIDL_C_INLINE_WRAPPERS
/*** IUnknown methods ***/
#define IWiaPropertyStorage_QueryInterface(This,riid,ppvObject) (This)->lpVtbl->QueryInterface(This,riid,ppvObject)
#define IWiaPropertyStorage_AddRef(This) (This)->lpVtbl->AddRef(This)
#define IWiaPropertyStorage_Release(This) (This)->lpVtbl->Release(This)
#else
/*** IUnknown methods ***/
static __WIDL_INLINE HRESULT IWiaPropertyStorage_QueryInterface(IWiaPropertyStorage* This,REFIID riid,void **ppvObject) {
    return This->lpVtbl->QueryInterface(This,riid,ppvObject);
}
static __WIDL_INLINE ULONG IWiaPropertyStorage_AddRef(IWiaPropertyStorage* This) {
    return This->lpVtbl->AddRef(This);
}
static __WIDL_INLINE ULONG IWiaPropertyStorage_Release(IWiaPropertyStorage* This) {
    return This->lpVtbl->Release(This);
}
#endif
#endif

#endif


#endif  /* __IWiaPropertyStorage_INTERFACE_DEFINED__ */

/*****************************************************************************
 * IWiaItem interface
 */
#ifndef __IWiaItem_INTERFACE_DEFINED__
#define __IWiaItem_INTERFACE_DEFINED__

DEFINE_GUID(IID_IWiaItem, 0x4db1ad10, 0x3391, 0x11d2, 0x9a,0x33, 0x00,0xc0,0x4f,0xa3,0x61,0x45);
#if defined(__cplusplus) && !defined(CINTERFACE)
MIDL_INTERFACE("4db1ad10-3391-11d2-9a33-00c04fa36145")
IWiaItem : public IUnknown
{
};
#ifdef __CRT_UUID_DECL
__CRT_UUID_DECL(IWiaItem, 0x4db1ad10, 0x3391, 0x11d2, 0x9a,0x33, 0x00,0xc0,0x4f,0xa3,0x61,0x45)
#endif
#else
typedef struct IWiaItemVtbl {
    BEGIN_INTERFACE

    /*** IUnknown methods ***/
    HRESULT (STDMETHODCALLTYPE *QueryInterface)(
        IWiaItem *This,
        REFIID riid,
        void **ppvObject);

    ULONG (STDMETHODCALLTYPE *AddRef)(
        IWiaItem *This);

    ULONG (STDMETHODCALLTYPE *Release)(
        IWiaItem *This);

    END_INTERFACE
} IWiaItemVtbl;

interface IWiaItem {
    CONST_VTBL IWiaItemVtbl* lpVtbl;
};

#ifdef COBJMACROS
#ifndef WIDL_C_INLINE_WRAPPERS
/*** IUnknown methods ***/
#define IWiaItem_QueryInterface(This,riid,ppvObject) (This)->lpVtbl->QueryInterface(This,riid,ppvObject)
#define IWiaItem_AddRef(This) (This)->lpVtbl->AddRef(This)
#define IWiaItem_Release(This) (This)->lpVtbl->Release(This)
#else
/*** IUnknown methods ***/
static __WIDL_INLINE HRESULT IWiaItem_QueryInterface(IWiaItem* This,REFIID riid,void **ppvObject) {
    return This->lpVtbl->QueryInterface(This,riid,ppvObject);
}
static __WIDL_INLINE ULONG IWiaItem_AddRef(IWiaItem* This) {
    return This->lpVtbl->AddRef(This);
}
static __WIDL_INLINE ULONG IWiaItem_Release(IWiaItem* This) {
    return This->lpVtbl->Release(This);
}
#endif
#endif

#endif


#endif  /* __IWiaItem_INTERFACE_DEFINED__ */

/*****************************************************************************
 * IWiaEventCallback interface
 */
#ifndef __IWiaEventCallback_INTERFACE_DEFINED__
#define __IWiaEventCallback_INTERFACE_DEFINED__

DEFINE_GUID(IID_IWiaEventCallback, 0xae6287b0, 0x0084, 0x11d2, 0x97,0x3b, 0x00,0xa0,0xc9,0x06,0x8f,0x2e);
#if defined(__cplusplus) && !defined(CINTERFACE)
MIDL_INTERFACE("ae6287b0-0084-11d2-973b-00a0c9068f2e")
IWiaEventCallback : public IUnknown
{
    virtual HRESULT STDMETHODCALLTYPE ImageEventCallback(
        const GUID *pEventGUID,
        BSTR bstrEventDescription,
        BSTR bstrDeviceID,
        BSTR bstrDeviceDescription,
        DWORD dwDeviceType,
        BSTR bstrFullItemName,
        ULONG *pulEventType,
        ULONG ulReserved) = 0;

};
#ifdef __CRT_UUID_DECL
__CRT_UUID_DECL(IWiaEventCallback, 0xae6287b0, 0x0084, 0x11d2, 0x97,0x3b, 0x00,0xa0,0xc9,0x06,0x8f,0x2e)
#endif
#else
typedef struct IWiaEventCallbackVtbl {
    BEGIN_INTERFACE

    /*** IUnknown methods ***/
    HRESULT (STDMETHODCALLTYPE *QueryInterface)(
        IWiaEventCallback *This,
        REFIID riid,
        void **ppvObject);

    ULONG (STDMETHODCALLTYPE *AddRef)(
        IWiaEventCallback *This);

    ULONG (STDMETHODCALLTYPE *Release)(
        IWiaEventCallback *This);

    /*** IWiaEventCallback methods ***/
    HRESULT (STDMETHODCALLTYPE *ImageEventCallback)(
        IWiaEventCallback *This,
        const GUID *pEventGUID,
        BSTR bstrEventDescription,
        BSTR bstrDeviceID,
        BSTR bstrDeviceDescription,
        DWORD dwDeviceType,
        BSTR bstrFullItemName,
        ULONG *pulEventType,
        ULONG ulReserved);

    END_INTERFACE
} IWiaEventCallbackVtbl;

interface IWiaEventCallback {
    CONST_VTBL IWiaEventCallbackVtbl* lpVtbl;
};

#ifdef COBJMACROS
#ifndef WIDL_C_INLINE_WRAPPERS
/*** IUnknown methods ***/
#define IWiaEventCallback_QueryInterface(This,riid,ppvObject) (This)->lpVtbl->QueryInterface(This,riid,ppvObject)
#define IWiaEventCallback_AddRef(This) (This)->lpVtbl->AddRef(This)
#define IWiaEventCallback_Release(This) (This)->lpVtbl->Release(This)
/*** IWiaEventCallback methods ***/
#define IWiaEventCallback_ImageEventCallback(This,pEventGUID,bstrEventDescription,bstrDeviceID,bstrDeviceDescription,dwDeviceType,bstrFullItemName,pulEventType,ulReserved) (This)->lpVtbl->ImageEventCallback(This,pEventGUID,bstrEventDescription,bstrDeviceID,bstrDeviceDescription,dwDeviceType,bstrFullItemName,pulEventType,ulReserved)
#else
/*** IUnknown methods ***/
static __WIDL_INLINE HRESULT IWiaEventCallback_QueryInterface(IWiaEventCallback* This,REFIID riid,void **ppvObject) {
    return This->lpVtbl->QueryInterface(This,riid,ppvObject);
}
static __WIDL_INLINE ULONG IWiaEventCallback_AddRef(IWiaEventCallback* This) {
    return This->lpVtbl->AddRef(This);
}
static __WIDL_INLINE ULONG IWiaEventCallback_Release(IWiaEventCallback* This) {
    return This->lpVtbl->Release(This);
}
/*** IWiaEventCallback methods ***/
static __WIDL_INLINE HRESULT IWiaEventCallback_ImageEventCallback(IWiaEventCallback* This,const GUID *pEventGUID,BSTR bstrEventDescription,BSTR bstrDeviceID,BSTR bstrDeviceDescription,DWORD dwDeviceType,BSTR bstrFullItemName,ULONG *pulEventType,ULONG ulReserved) {
    return This->lpVtbl->ImageEventCallback(This,pEventGUID,bstrEventDescription,bstrDeviceID,bstrDeviceDescription,dwDeviceType,bstrFullItemName,pulEventType,ulReserved);
}
#endif
#endif

#endif


#endif  /* __IWiaEventCallback_INTERFACE_DEFINED__ */

/* Begin additional prototypes for all interfaces */

ULONG           __RPC_USER BSTR_UserSize     (ULONG *, ULONG, BSTR *);
unsigned char * __RPC_USER BSTR_UserMarshal  (ULONG *, unsigned char *, BSTR *);
unsigned char * __RPC_USER BSTR_UserUnmarshal(ULONG *, unsigned char *, BSTR *);
void            __RPC_USER BSTR_UserFree     (ULONG *, BSTR *);
ULONG           __RPC_USER HWND_UserSize     (ULONG *, ULONG, HWND *);
unsigned char * __RPC_USER HWND_UserMarshal  (ULONG *, unsigned char *, HWND *);
unsigned char * __RPC_USER HWND_UserUnmarshal(ULONG *, unsigned char *, HWND *);
void            __RPC_USER HWND_UserFree     (ULONG *, HWND *);

/* End additional prototypes */

#ifdef __cplusplus
}
#endif

#endif /* __wia_xp_h__ */