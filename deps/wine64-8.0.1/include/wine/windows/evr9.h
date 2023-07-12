/*** Autogenerated by WIDL 8.0.1 from ../include/evr9.idl - Do not edit ***/

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

#ifndef __evr9_h__
#define __evr9_h__

#ifndef __WIDL_INLINE
#if defined(__cplusplus) || defined(_MSC_VER)
#define __WIDL_INLINE inline
#elif defined(__GNUC__)
#define __WIDL_INLINE __inline__
#endif
#endif

/* Forward declarations */

#ifndef __IMFVideoMixerBitmap_FWD_DEFINED__
#define __IMFVideoMixerBitmap_FWD_DEFINED__
typedef interface IMFVideoMixerBitmap IMFVideoMixerBitmap;
#ifdef __cplusplus
interface IMFVideoMixerBitmap;
#endif /* __cplusplus */
#endif

#ifndef __IMFVideoProcessor_FWD_DEFINED__
#define __IMFVideoProcessor_FWD_DEFINED__
typedef interface IMFVideoProcessor IMFVideoProcessor;
#ifdef __cplusplus
interface IMFVideoProcessor;
#endif /* __cplusplus */
#endif

/* Headers for imported files */

#include <unknwn.h>
#include <evr.h>
#include <dxva2api.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct MFVideoAlphaBitmapParams {
    DWORD dwFlags;
    COLORREF clrSrcKey;
    RECT rcSrc;
    MFVideoNormalizedRect nrcDest;
    FLOAT fAlpha;
    DWORD dwFilterMode;
} MFVideoAlphaBitmapParams;
typedef struct MFVideoAlphaBitmap {
    BOOL GetBitmapFromDC;
    union {
        HDC hdc;
        IDirect3DSurface9 *pDDS;
    } bitmap;
    MFVideoAlphaBitmapParams params;
} MFVideoAlphaBitmap;
typedef enum __WIDL_evr9_generated_name_00000033 {
    MFVideoAlphaBitmap_EntireDDS = 0x1,
    MFVideoAlphaBitmap_SrcColorKey = 0x2,
    MFVideoAlphaBitmap_SrcRect = 0x4,
    MFVideoAlphaBitmap_DestRect = 0x8,
    MFVideoAlphaBitmap_FilterMode = 0x10,
    MFVideoAlphaBitmap_Alpha = 0x20,
    MFVideoAlphaBitmap_BitMask = 0x3f
} MFVideoAlphaBitmapFlags;
/*****************************************************************************
 * IMFVideoMixerBitmap interface
 */
#ifndef __IMFVideoMixerBitmap_INTERFACE_DEFINED__
#define __IMFVideoMixerBitmap_INTERFACE_DEFINED__

DEFINE_GUID(IID_IMFVideoMixerBitmap, 0x814c7b20, 0x0fdb, 0x4eec, 0xaf,0x8f, 0xf9,0x57,0xc8,0xf6,0x9e,0xdc);
#if defined(__cplusplus) && !defined(CINTERFACE)
MIDL_INTERFACE("814c7b20-0fdb-4eec-af8f-f957c8f69edc")
IMFVideoMixerBitmap : public IUnknown
{
    virtual HRESULT STDMETHODCALLTYPE SetAlphaBitmap(
        const MFVideoAlphaBitmap *bitmap) = 0;

    virtual HRESULT STDMETHODCALLTYPE ClearAlphaBitmap(
        ) = 0;

    virtual HRESULT STDMETHODCALLTYPE UpdateAlphaBitmapParameters(
        const MFVideoAlphaBitmapParams *params) = 0;

    virtual HRESULT STDMETHODCALLTYPE GetAlphaBitmapParameters(
        MFVideoAlphaBitmapParams *params) = 0;

};
#ifdef __CRT_UUID_DECL
__CRT_UUID_DECL(IMFVideoMixerBitmap, 0x814c7b20, 0x0fdb, 0x4eec, 0xaf,0x8f, 0xf9,0x57,0xc8,0xf6,0x9e,0xdc)
#endif
#else
typedef struct IMFVideoMixerBitmapVtbl {
    BEGIN_INTERFACE

    /*** IUnknown methods ***/
    HRESULT (STDMETHODCALLTYPE *QueryInterface)(
        IMFVideoMixerBitmap *This,
        REFIID riid,
        void **ppvObject);

    ULONG (STDMETHODCALLTYPE *AddRef)(
        IMFVideoMixerBitmap *This);

    ULONG (STDMETHODCALLTYPE *Release)(
        IMFVideoMixerBitmap *This);

    /*** IMFVideoMixerBitmap methods ***/
    HRESULT (STDMETHODCALLTYPE *SetAlphaBitmap)(
        IMFVideoMixerBitmap *This,
        const MFVideoAlphaBitmap *bitmap);

    HRESULT (STDMETHODCALLTYPE *ClearAlphaBitmap)(
        IMFVideoMixerBitmap *This);

    HRESULT (STDMETHODCALLTYPE *UpdateAlphaBitmapParameters)(
        IMFVideoMixerBitmap *This,
        const MFVideoAlphaBitmapParams *params);

    HRESULT (STDMETHODCALLTYPE *GetAlphaBitmapParameters)(
        IMFVideoMixerBitmap *This,
        MFVideoAlphaBitmapParams *params);

    END_INTERFACE
} IMFVideoMixerBitmapVtbl;

interface IMFVideoMixerBitmap {
    CONST_VTBL IMFVideoMixerBitmapVtbl* lpVtbl;
};

#ifdef COBJMACROS
#ifndef WIDL_C_INLINE_WRAPPERS
/*** IUnknown methods ***/
#define IMFVideoMixerBitmap_QueryInterface(This,riid,ppvObject) (This)->lpVtbl->QueryInterface(This,riid,ppvObject)
#define IMFVideoMixerBitmap_AddRef(This) (This)->lpVtbl->AddRef(This)
#define IMFVideoMixerBitmap_Release(This) (This)->lpVtbl->Release(This)
/*** IMFVideoMixerBitmap methods ***/
#define IMFVideoMixerBitmap_SetAlphaBitmap(This,bitmap) (This)->lpVtbl->SetAlphaBitmap(This,bitmap)
#define IMFVideoMixerBitmap_ClearAlphaBitmap(This) (This)->lpVtbl->ClearAlphaBitmap(This)
#define IMFVideoMixerBitmap_UpdateAlphaBitmapParameters(This,params) (This)->lpVtbl->UpdateAlphaBitmapParameters(This,params)
#define IMFVideoMixerBitmap_GetAlphaBitmapParameters(This,params) (This)->lpVtbl->GetAlphaBitmapParameters(This,params)
#else
/*** IUnknown methods ***/
static __WIDL_INLINE HRESULT IMFVideoMixerBitmap_QueryInterface(IMFVideoMixerBitmap* This,REFIID riid,void **ppvObject) {
    return This->lpVtbl->QueryInterface(This,riid,ppvObject);
}
static __WIDL_INLINE ULONG IMFVideoMixerBitmap_AddRef(IMFVideoMixerBitmap* This) {
    return This->lpVtbl->AddRef(This);
}
static __WIDL_INLINE ULONG IMFVideoMixerBitmap_Release(IMFVideoMixerBitmap* This) {
    return This->lpVtbl->Release(This);
}
/*** IMFVideoMixerBitmap methods ***/
static __WIDL_INLINE HRESULT IMFVideoMixerBitmap_SetAlphaBitmap(IMFVideoMixerBitmap* This,const MFVideoAlphaBitmap *bitmap) {
    return This->lpVtbl->SetAlphaBitmap(This,bitmap);
}
static __WIDL_INLINE HRESULT IMFVideoMixerBitmap_ClearAlphaBitmap(IMFVideoMixerBitmap* This) {
    return This->lpVtbl->ClearAlphaBitmap(This);
}
static __WIDL_INLINE HRESULT IMFVideoMixerBitmap_UpdateAlphaBitmapParameters(IMFVideoMixerBitmap* This,const MFVideoAlphaBitmapParams *params) {
    return This->lpVtbl->UpdateAlphaBitmapParameters(This,params);
}
static __WIDL_INLINE HRESULT IMFVideoMixerBitmap_GetAlphaBitmapParameters(IMFVideoMixerBitmap* This,MFVideoAlphaBitmapParams *params) {
    return This->lpVtbl->GetAlphaBitmapParameters(This,params);
}
#endif
#endif

#endif


#endif  /* __IMFVideoMixerBitmap_INTERFACE_DEFINED__ */

/*****************************************************************************
 * IMFVideoProcessor interface
 */
#ifndef __IMFVideoProcessor_INTERFACE_DEFINED__
#define __IMFVideoProcessor_INTERFACE_DEFINED__

DEFINE_GUID(IID_IMFVideoProcessor, 0x6ab0000c, 0xfece, 0x4d1f, 0xa2,0xac, 0xa9,0x57,0x35,0x30,0x65,0x6e);
#if defined(__cplusplus) && !defined(CINTERFACE)
MIDL_INTERFACE("6ab0000c-fece-4d1f-a2ac-a9573530656e")
IMFVideoProcessor : public IUnknown
{
    virtual HRESULT STDMETHODCALLTYPE GetAvailableVideoProcessorModes(
        UINT *count,
        GUID **modes) = 0;

    virtual HRESULT STDMETHODCALLTYPE GetVideoProcessorCaps(
        GUID *mode,
        DXVA2_VideoProcessorCaps *caps) = 0;

    virtual HRESULT STDMETHODCALLTYPE GetVideoProcessorMode(
        GUID *mode) = 0;

    virtual HRESULT STDMETHODCALLTYPE SetVideoProcessorMode(
        GUID *mode) = 0;

    virtual HRESULT STDMETHODCALLTYPE GetProcAmpRange(
        DWORD prop,
        DXVA2_ValueRange *range) = 0;

    virtual HRESULT STDMETHODCALLTYPE GetProcAmpValues(
        DWORD flags,
        DXVA2_ProcAmpValues *values) = 0;

    virtual HRESULT STDMETHODCALLTYPE SetProcAmpValues(
        DWORD flags,
        DXVA2_ProcAmpValues *values) = 0;

    virtual HRESULT STDMETHODCALLTYPE GetFilteringRange(
        DWORD prop,
        DXVA2_ValueRange *range) = 0;

    virtual HRESULT STDMETHODCALLTYPE GetFilteringValue(
        DWORD prop,
        DXVA2_Fixed32 *value) = 0;

    virtual HRESULT STDMETHODCALLTYPE SetFilteringValue(
        DWORD prop,
        DXVA2_Fixed32 *value) = 0;

    virtual HRESULT STDMETHODCALLTYPE GetBackgroundColor(
        COLORREF *color) = 0;

    virtual HRESULT STDMETHODCALLTYPE SetBackgroundColor(
        COLORREF color) = 0;

};
#ifdef __CRT_UUID_DECL
__CRT_UUID_DECL(IMFVideoProcessor, 0x6ab0000c, 0xfece, 0x4d1f, 0xa2,0xac, 0xa9,0x57,0x35,0x30,0x65,0x6e)
#endif
#else
typedef struct IMFVideoProcessorVtbl {
    BEGIN_INTERFACE

    /*** IUnknown methods ***/
    HRESULT (STDMETHODCALLTYPE *QueryInterface)(
        IMFVideoProcessor *This,
        REFIID riid,
        void **ppvObject);

    ULONG (STDMETHODCALLTYPE *AddRef)(
        IMFVideoProcessor *This);

    ULONG (STDMETHODCALLTYPE *Release)(
        IMFVideoProcessor *This);

    /*** IMFVideoProcessor methods ***/
    HRESULT (STDMETHODCALLTYPE *GetAvailableVideoProcessorModes)(
        IMFVideoProcessor *This,
        UINT *count,
        GUID **modes);

    HRESULT (STDMETHODCALLTYPE *GetVideoProcessorCaps)(
        IMFVideoProcessor *This,
        GUID *mode,
        DXVA2_VideoProcessorCaps *caps);

    HRESULT (STDMETHODCALLTYPE *GetVideoProcessorMode)(
        IMFVideoProcessor *This,
        GUID *mode);

    HRESULT (STDMETHODCALLTYPE *SetVideoProcessorMode)(
        IMFVideoProcessor *This,
        GUID *mode);

    HRESULT (STDMETHODCALLTYPE *GetProcAmpRange)(
        IMFVideoProcessor *This,
        DWORD prop,
        DXVA2_ValueRange *range);

    HRESULT (STDMETHODCALLTYPE *GetProcAmpValues)(
        IMFVideoProcessor *This,
        DWORD flags,
        DXVA2_ProcAmpValues *values);

    HRESULT (STDMETHODCALLTYPE *SetProcAmpValues)(
        IMFVideoProcessor *This,
        DWORD flags,
        DXVA2_ProcAmpValues *values);

    HRESULT (STDMETHODCALLTYPE *GetFilteringRange)(
        IMFVideoProcessor *This,
        DWORD prop,
        DXVA2_ValueRange *range);

    HRESULT (STDMETHODCALLTYPE *GetFilteringValue)(
        IMFVideoProcessor *This,
        DWORD prop,
        DXVA2_Fixed32 *value);

    HRESULT (STDMETHODCALLTYPE *SetFilteringValue)(
        IMFVideoProcessor *This,
        DWORD prop,
        DXVA2_Fixed32 *value);

    HRESULT (STDMETHODCALLTYPE *GetBackgroundColor)(
        IMFVideoProcessor *This,
        COLORREF *color);

    HRESULT (STDMETHODCALLTYPE *SetBackgroundColor)(
        IMFVideoProcessor *This,
        COLORREF color);

    END_INTERFACE
} IMFVideoProcessorVtbl;

interface IMFVideoProcessor {
    CONST_VTBL IMFVideoProcessorVtbl* lpVtbl;
};

#ifdef COBJMACROS
#ifndef WIDL_C_INLINE_WRAPPERS
/*** IUnknown methods ***/
#define IMFVideoProcessor_QueryInterface(This,riid,ppvObject) (This)->lpVtbl->QueryInterface(This,riid,ppvObject)
#define IMFVideoProcessor_AddRef(This) (This)->lpVtbl->AddRef(This)
#define IMFVideoProcessor_Release(This) (This)->lpVtbl->Release(This)
/*** IMFVideoProcessor methods ***/
#define IMFVideoProcessor_GetAvailableVideoProcessorModes(This,count,modes) (This)->lpVtbl->GetAvailableVideoProcessorModes(This,count,modes)
#define IMFVideoProcessor_GetVideoProcessorCaps(This,mode,caps) (This)->lpVtbl->GetVideoProcessorCaps(This,mode,caps)
#define IMFVideoProcessor_GetVideoProcessorMode(This,mode) (This)->lpVtbl->GetVideoProcessorMode(This,mode)
#define IMFVideoProcessor_SetVideoProcessorMode(This,mode) (This)->lpVtbl->SetVideoProcessorMode(This,mode)
#define IMFVideoProcessor_GetProcAmpRange(This,prop,range) (This)->lpVtbl->GetProcAmpRange(This,prop,range)
#define IMFVideoProcessor_GetProcAmpValues(This,flags,values) (This)->lpVtbl->GetProcAmpValues(This,flags,values)
#define IMFVideoProcessor_SetProcAmpValues(This,flags,values) (This)->lpVtbl->SetProcAmpValues(This,flags,values)
#define IMFVideoProcessor_GetFilteringRange(This,prop,range) (This)->lpVtbl->GetFilteringRange(This,prop,range)
#define IMFVideoProcessor_GetFilteringValue(This,prop,value) (This)->lpVtbl->GetFilteringValue(This,prop,value)
#define IMFVideoProcessor_SetFilteringValue(This,prop,value) (This)->lpVtbl->SetFilteringValue(This,prop,value)
#define IMFVideoProcessor_GetBackgroundColor(This,color) (This)->lpVtbl->GetBackgroundColor(This,color)
#define IMFVideoProcessor_SetBackgroundColor(This,color) (This)->lpVtbl->SetBackgroundColor(This,color)
#else
/*** IUnknown methods ***/
static __WIDL_INLINE HRESULT IMFVideoProcessor_QueryInterface(IMFVideoProcessor* This,REFIID riid,void **ppvObject) {
    return This->lpVtbl->QueryInterface(This,riid,ppvObject);
}
static __WIDL_INLINE ULONG IMFVideoProcessor_AddRef(IMFVideoProcessor* This) {
    return This->lpVtbl->AddRef(This);
}
static __WIDL_INLINE ULONG IMFVideoProcessor_Release(IMFVideoProcessor* This) {
    return This->lpVtbl->Release(This);
}
/*** IMFVideoProcessor methods ***/
static __WIDL_INLINE HRESULT IMFVideoProcessor_GetAvailableVideoProcessorModes(IMFVideoProcessor* This,UINT *count,GUID **modes) {
    return This->lpVtbl->GetAvailableVideoProcessorModes(This,count,modes);
}
static __WIDL_INLINE HRESULT IMFVideoProcessor_GetVideoProcessorCaps(IMFVideoProcessor* This,GUID *mode,DXVA2_VideoProcessorCaps *caps) {
    return This->lpVtbl->GetVideoProcessorCaps(This,mode,caps);
}
static __WIDL_INLINE HRESULT IMFVideoProcessor_GetVideoProcessorMode(IMFVideoProcessor* This,GUID *mode) {
    return This->lpVtbl->GetVideoProcessorMode(This,mode);
}
static __WIDL_INLINE HRESULT IMFVideoProcessor_SetVideoProcessorMode(IMFVideoProcessor* This,GUID *mode) {
    return This->lpVtbl->SetVideoProcessorMode(This,mode);
}
static __WIDL_INLINE HRESULT IMFVideoProcessor_GetProcAmpRange(IMFVideoProcessor* This,DWORD prop,DXVA2_ValueRange *range) {
    return This->lpVtbl->GetProcAmpRange(This,prop,range);
}
static __WIDL_INLINE HRESULT IMFVideoProcessor_GetProcAmpValues(IMFVideoProcessor* This,DWORD flags,DXVA2_ProcAmpValues *values) {
    return This->lpVtbl->GetProcAmpValues(This,flags,values);
}
static __WIDL_INLINE HRESULT IMFVideoProcessor_SetProcAmpValues(IMFVideoProcessor* This,DWORD flags,DXVA2_ProcAmpValues *values) {
    return This->lpVtbl->SetProcAmpValues(This,flags,values);
}
static __WIDL_INLINE HRESULT IMFVideoProcessor_GetFilteringRange(IMFVideoProcessor* This,DWORD prop,DXVA2_ValueRange *range) {
    return This->lpVtbl->GetFilteringRange(This,prop,range);
}
static __WIDL_INLINE HRESULT IMFVideoProcessor_GetFilteringValue(IMFVideoProcessor* This,DWORD prop,DXVA2_Fixed32 *value) {
    return This->lpVtbl->GetFilteringValue(This,prop,value);
}
static __WIDL_INLINE HRESULT IMFVideoProcessor_SetFilteringValue(IMFVideoProcessor* This,DWORD prop,DXVA2_Fixed32 *value) {
    return This->lpVtbl->SetFilteringValue(This,prop,value);
}
static __WIDL_INLINE HRESULT IMFVideoProcessor_GetBackgroundColor(IMFVideoProcessor* This,COLORREF *color) {
    return This->lpVtbl->GetBackgroundColor(This,color);
}
static __WIDL_INLINE HRESULT IMFVideoProcessor_SetBackgroundColor(IMFVideoProcessor* This,COLORREF color) {
    return This->lpVtbl->SetBackgroundColor(This,color);
}
#endif
#endif

#endif


#endif  /* __IMFVideoProcessor_INTERFACE_DEFINED__ */

/* Begin additional prototypes for all interfaces */


/* End additional prototypes */

#ifdef __cplusplus
}
#endif

#endif /* __evr9_h__ */
