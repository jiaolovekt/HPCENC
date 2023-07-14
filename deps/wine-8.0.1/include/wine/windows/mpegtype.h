/*** Autogenerated by WIDL 8.0.1 from ../include/mpegtype.idl - Do not edit ***/

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

#ifndef __mpegtype_h__
#define __mpegtype_h__

#ifndef __WIDL_INLINE
#if defined(__cplusplus) || defined(_MSC_VER)
#define __WIDL_INLINE inline
#elif defined(__GNUC__)
#define __WIDL_INLINE __inline__
#endif
#endif

/* Forward declarations */

#ifndef __IMpegAudioDecoder_FWD_DEFINED__
#define __IMpegAudioDecoder_FWD_DEFINED__
typedef interface IMpegAudioDecoder IMpegAudioDecoder;
#ifdef __cplusplus
interface IMpegAudioDecoder;
#endif /* __cplusplus */
#endif

/* Headers for imported files */

#include <unknwn.h>

#ifdef __cplusplus
extern "C" {
#endif

#include <mmreg.h>
#if 0
typedef struct tWAVEFORMATEX {
    WORD wFormatTag;
    WORD nChannels;
    DWORD nSamplesPerSec;
    DWORD nAvgBytesPerSec;
    WORD nBlockAlign;
    WORD wBitsPerSample;
    WORD cbSize;
    BYTE pExtraBytes[1];
} WAVEFORMATEX;
typedef struct tWAVEFORMATEX *PWAVEFORMATEX;
typedef struct tWAVEFORMATEX *NPWAVEFORMATEX;
typedef struct tWAVEFORMATEX *LPWAVEFORMATEX;
typedef struct __WIDL_mpegtype_generated_name_0000000B {
    WORD wFormatTag;
    WORD nChannels;
    DWORD nSamplesPerSec;
    DWORD nAvgBytesPerSec;
    WORD nBlockAlign;
    WORD wBitsPerSample;
    WORD cbSize;
    WORD wValidBitsPerSample;
    DWORD dwChannelMask;
    GUID SubFormat;
} WAVEFORMATEXTENSIBLE;
typedef struct __WIDL_mpegtype_generated_name_0000000B *PWAVEFORMATEXTENSIBLE;
typedef struct __WIDL_mpegtype_generated_name_0000000C {
    WAVEFORMATEX wfx;
    WORD fwHeadLayer;
    DWORD dwHeadBitrate;
    WORD fwHeadMode;
    WORD fwHeadModeExt;
    WORD wHeadEmphasis;
    WORD fwHeadFlags;
    DWORD dwPTSLow;
    DWORD dwPTSHigh;
} MPEG1WAVEFORMAT;
#endif
/*****************************************************************************
 * IMpegAudioDecoder interface
 */
#ifndef __IMpegAudioDecoder_INTERFACE_DEFINED__
#define __IMpegAudioDecoder_INTERFACE_DEFINED__

#if defined(__cplusplus) && !defined(CINTERFACE)
interface IMpegAudioDecoder : public IUnknown
{
    virtual HRESULT STDMETHODCALLTYPE get_FrequencyDivider(
        ULONG *divider) = 0;

    virtual HRESULT STDMETHODCALLTYPE put_FrequencyDivider(
        ULONG divider) = 0;

    virtual HRESULT STDMETHODCALLTYPE get_DecoderAccuracy(
        ULONG *accuracy) = 0;

    virtual HRESULT STDMETHODCALLTYPE put_DecoderAccuracy(
        ULONG accuracy) = 0;

    virtual HRESULT STDMETHODCALLTYPE get_Stereo(
        ULONG *stereo) = 0;

    virtual HRESULT STDMETHODCALLTYPE put_Stereo(
        ULONG stereo) = 0;

    virtual HRESULT STDMETHODCALLTYPE get_DecoderWordSize(
        ULONG *word_size) = 0;

    virtual HRESULT STDMETHODCALLTYPE put_DecoderWordSize(
        ULONG word_size) = 0;

    virtual HRESULT STDMETHODCALLTYPE get_IntegerDecode(
        ULONG *integer_decode) = 0;

    virtual HRESULT STDMETHODCALLTYPE put_IntegerDecode(
        ULONG integer_decode) = 0;

    virtual HRESULT STDMETHODCALLTYPE get_DualMode(
        ULONG *dual_mode) = 0;

    virtual HRESULT STDMETHODCALLTYPE put_DualMode(
        ULONG dual_mode) = 0;

    virtual HRESULT STDMETHODCALLTYPE get_AudioFormat(
        MPEG1WAVEFORMAT *format) = 0;

};
#else
typedef struct IMpegAudioDecoderVtbl {
    BEGIN_INTERFACE

    /*** IUnknown methods ***/
    HRESULT (STDMETHODCALLTYPE *QueryInterface)(
        IMpegAudioDecoder *This,
        REFIID riid,
        void **ppvObject);

    ULONG (STDMETHODCALLTYPE *AddRef)(
        IMpegAudioDecoder *This);

    ULONG (STDMETHODCALLTYPE *Release)(
        IMpegAudioDecoder *This);

    /*** IMpegAudioDecoder methods ***/
    HRESULT (STDMETHODCALLTYPE *get_FrequencyDivider)(
        IMpegAudioDecoder *This,
        ULONG *divider);

    HRESULT (STDMETHODCALLTYPE *put_FrequencyDivider)(
        IMpegAudioDecoder *This,
        ULONG divider);

    HRESULT (STDMETHODCALLTYPE *get_DecoderAccuracy)(
        IMpegAudioDecoder *This,
        ULONG *accuracy);

    HRESULT (STDMETHODCALLTYPE *put_DecoderAccuracy)(
        IMpegAudioDecoder *This,
        ULONG accuracy);

    HRESULT (STDMETHODCALLTYPE *get_Stereo)(
        IMpegAudioDecoder *This,
        ULONG *stereo);

    HRESULT (STDMETHODCALLTYPE *put_Stereo)(
        IMpegAudioDecoder *This,
        ULONG stereo);

    HRESULT (STDMETHODCALLTYPE *get_DecoderWordSize)(
        IMpegAudioDecoder *This,
        ULONG *word_size);

    HRESULT (STDMETHODCALLTYPE *put_DecoderWordSize)(
        IMpegAudioDecoder *This,
        ULONG word_size);

    HRESULT (STDMETHODCALLTYPE *get_IntegerDecode)(
        IMpegAudioDecoder *This,
        ULONG *integer_decode);

    HRESULT (STDMETHODCALLTYPE *put_IntegerDecode)(
        IMpegAudioDecoder *This,
        ULONG integer_decode);

    HRESULT (STDMETHODCALLTYPE *get_DualMode)(
        IMpegAudioDecoder *This,
        ULONG *dual_mode);

    HRESULT (STDMETHODCALLTYPE *put_DualMode)(
        IMpegAudioDecoder *This,
        ULONG dual_mode);

    HRESULT (STDMETHODCALLTYPE *get_AudioFormat)(
        IMpegAudioDecoder *This,
        MPEG1WAVEFORMAT *format);

    END_INTERFACE
} IMpegAudioDecoderVtbl;

interface IMpegAudioDecoder {
    CONST_VTBL IMpegAudioDecoderVtbl* lpVtbl;
};

#ifdef COBJMACROS
#ifndef WIDL_C_INLINE_WRAPPERS
/*** IUnknown methods ***/
#define IMpegAudioDecoder_QueryInterface(This,riid,ppvObject) (This)->lpVtbl->QueryInterface(This,riid,ppvObject)
#define IMpegAudioDecoder_AddRef(This) (This)->lpVtbl->AddRef(This)
#define IMpegAudioDecoder_Release(This) (This)->lpVtbl->Release(This)
/*** IMpegAudioDecoder methods ***/
#define IMpegAudioDecoder_get_FrequencyDivider(This,divider) (This)->lpVtbl->get_FrequencyDivider(This,divider)
#define IMpegAudioDecoder_put_FrequencyDivider(This,divider) (This)->lpVtbl->put_FrequencyDivider(This,divider)
#define IMpegAudioDecoder_get_DecoderAccuracy(This,accuracy) (This)->lpVtbl->get_DecoderAccuracy(This,accuracy)
#define IMpegAudioDecoder_put_DecoderAccuracy(This,accuracy) (This)->lpVtbl->put_DecoderAccuracy(This,accuracy)
#define IMpegAudioDecoder_get_Stereo(This,stereo) (This)->lpVtbl->get_Stereo(This,stereo)
#define IMpegAudioDecoder_put_Stereo(This,stereo) (This)->lpVtbl->put_Stereo(This,stereo)
#define IMpegAudioDecoder_get_DecoderWordSize(This,word_size) (This)->lpVtbl->get_DecoderWordSize(This,word_size)
#define IMpegAudioDecoder_put_DecoderWordSize(This,word_size) (This)->lpVtbl->put_DecoderWordSize(This,word_size)
#define IMpegAudioDecoder_get_IntegerDecode(This,integer_decode) (This)->lpVtbl->get_IntegerDecode(This,integer_decode)
#define IMpegAudioDecoder_put_IntegerDecode(This,integer_decode) (This)->lpVtbl->put_IntegerDecode(This,integer_decode)
#define IMpegAudioDecoder_get_DualMode(This,dual_mode) (This)->lpVtbl->get_DualMode(This,dual_mode)
#define IMpegAudioDecoder_put_DualMode(This,dual_mode) (This)->lpVtbl->put_DualMode(This,dual_mode)
#define IMpegAudioDecoder_get_AudioFormat(This,format) (This)->lpVtbl->get_AudioFormat(This,format)
#else
/*** IUnknown methods ***/
static __WIDL_INLINE HRESULT IMpegAudioDecoder_QueryInterface(IMpegAudioDecoder* This,REFIID riid,void **ppvObject) {
    return This->lpVtbl->QueryInterface(This,riid,ppvObject);
}
static __WIDL_INLINE ULONG IMpegAudioDecoder_AddRef(IMpegAudioDecoder* This) {
    return This->lpVtbl->AddRef(This);
}
static __WIDL_INLINE ULONG IMpegAudioDecoder_Release(IMpegAudioDecoder* This) {
    return This->lpVtbl->Release(This);
}
/*** IMpegAudioDecoder methods ***/
static __WIDL_INLINE HRESULT IMpegAudioDecoder_get_FrequencyDivider(IMpegAudioDecoder* This,ULONG *divider) {
    return This->lpVtbl->get_FrequencyDivider(This,divider);
}
static __WIDL_INLINE HRESULT IMpegAudioDecoder_put_FrequencyDivider(IMpegAudioDecoder* This,ULONG divider) {
    return This->lpVtbl->put_FrequencyDivider(This,divider);
}
static __WIDL_INLINE HRESULT IMpegAudioDecoder_get_DecoderAccuracy(IMpegAudioDecoder* This,ULONG *accuracy) {
    return This->lpVtbl->get_DecoderAccuracy(This,accuracy);
}
static __WIDL_INLINE HRESULT IMpegAudioDecoder_put_DecoderAccuracy(IMpegAudioDecoder* This,ULONG accuracy) {
    return This->lpVtbl->put_DecoderAccuracy(This,accuracy);
}
static __WIDL_INLINE HRESULT IMpegAudioDecoder_get_Stereo(IMpegAudioDecoder* This,ULONG *stereo) {
    return This->lpVtbl->get_Stereo(This,stereo);
}
static __WIDL_INLINE HRESULT IMpegAudioDecoder_put_Stereo(IMpegAudioDecoder* This,ULONG stereo) {
    return This->lpVtbl->put_Stereo(This,stereo);
}
static __WIDL_INLINE HRESULT IMpegAudioDecoder_get_DecoderWordSize(IMpegAudioDecoder* This,ULONG *word_size) {
    return This->lpVtbl->get_DecoderWordSize(This,word_size);
}
static __WIDL_INLINE HRESULT IMpegAudioDecoder_put_DecoderWordSize(IMpegAudioDecoder* This,ULONG word_size) {
    return This->lpVtbl->put_DecoderWordSize(This,word_size);
}
static __WIDL_INLINE HRESULT IMpegAudioDecoder_get_IntegerDecode(IMpegAudioDecoder* This,ULONG *integer_decode) {
    return This->lpVtbl->get_IntegerDecode(This,integer_decode);
}
static __WIDL_INLINE HRESULT IMpegAudioDecoder_put_IntegerDecode(IMpegAudioDecoder* This,ULONG integer_decode) {
    return This->lpVtbl->put_IntegerDecode(This,integer_decode);
}
static __WIDL_INLINE HRESULT IMpegAudioDecoder_get_DualMode(IMpegAudioDecoder* This,ULONG *dual_mode) {
    return This->lpVtbl->get_DualMode(This,dual_mode);
}
static __WIDL_INLINE HRESULT IMpegAudioDecoder_put_DualMode(IMpegAudioDecoder* This,ULONG dual_mode) {
    return This->lpVtbl->put_DualMode(This,dual_mode);
}
static __WIDL_INLINE HRESULT IMpegAudioDecoder_get_AudioFormat(IMpegAudioDecoder* This,MPEG1WAVEFORMAT *format) {
    return This->lpVtbl->get_AudioFormat(This,format);
}
#endif
#endif

#endif


#endif  /* __IMpegAudioDecoder_INTERFACE_DEFINED__ */

/* Begin additional prototypes for all interfaces */


/* End additional prototypes */

#ifdef __cplusplus
}
#endif

#endif /* __mpegtype_h__ */
