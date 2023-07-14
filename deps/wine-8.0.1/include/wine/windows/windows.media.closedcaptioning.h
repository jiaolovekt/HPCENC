/*** Autogenerated by WIDL 8.0.1 from ../include/windows.media.closedcaptioning.idl - Do not edit ***/

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

#ifndef __windows_media_closedcaptioning_h__
#define __windows_media_closedcaptioning_h__

#ifndef __WIDL_INLINE
#if defined(__cplusplus) || defined(_MSC_VER)
#define __WIDL_INLINE inline
#elif defined(__GNUC__)
#define __WIDL_INLINE __inline__
#endif
#endif

/* Forward declarations */

#ifndef ____x_ABI_CWindows_CMedia_CClosedCaptioning_CIClosedCaptionPropertiesStatics_FWD_DEFINED__
#define ____x_ABI_CWindows_CMedia_CClosedCaptioning_CIClosedCaptionPropertiesStatics_FWD_DEFINED__
typedef interface __x_ABI_CWindows_CMedia_CClosedCaptioning_CIClosedCaptionPropertiesStatics __x_ABI_CWindows_CMedia_CClosedCaptioning_CIClosedCaptionPropertiesStatics;
#ifdef __cplusplus
#define __x_ABI_CWindows_CMedia_CClosedCaptioning_CIClosedCaptionPropertiesStatics ABI::Windows::Media::ClosedCaptioning::IClosedCaptionPropertiesStatics
namespace ABI {
    namespace Windows {
        namespace Media {
            namespace ClosedCaptioning {
                interface IClosedCaptionPropertiesStatics;
            }
        }
    }
}
#endif /* __cplusplus */
#endif

#ifndef ____x_ABI_CWindows_CMedia_CClosedCaptioning_CClosedCaptionProperties_FWD_DEFINED__
#define ____x_ABI_CWindows_CMedia_CClosedCaptioning_CClosedCaptionProperties_FWD_DEFINED__
#ifdef __cplusplus
namespace ABI {
    namespace Windows {
        namespace Media {
            namespace ClosedCaptioning {
                class ClosedCaptionProperties;
            }
        }
    }
}
#else
typedef struct __x_ABI_CWindows_CMedia_CClosedCaptioning_CClosedCaptionProperties __x_ABI_CWindows_CMedia_CClosedCaptioning_CClosedCaptionProperties;
#endif /* defined __cplusplus */
#endif /* defined ____x_ABI_CWindows_CMedia_CClosedCaptioning_CClosedCaptionProperties_FWD_DEFINED__ */

/* Headers for imported files */

#include <inspectable.h>
#include <asyncinfo.h>
#include <eventtoken.h>
#include <windowscontracts.h>
#include <windows.foundation.h>
#include <windows.ui.h>

#ifdef __cplusplus
extern "C" {
#endif

#ifndef __cplusplus
typedef enum __x_ABI_CWindows_CMedia_CClosedCaptioning_CClosedCaptionColor __x_ABI_CWindows_CMedia_CClosedCaptioning_CClosedCaptionColor;
#endif /* __cplusplus */

#ifndef __cplusplus
typedef enum __x_ABI_CWindows_CMedia_CClosedCaptioning_CClosedCaptionEdgeEffect __x_ABI_CWindows_CMedia_CClosedCaptioning_CClosedCaptionEdgeEffect;
#endif /* __cplusplus */

#ifndef __cplusplus
typedef enum __x_ABI_CWindows_CMedia_CClosedCaptioning_CClosedCaptionOpacity __x_ABI_CWindows_CMedia_CClosedCaptioning_CClosedCaptionOpacity;
#endif /* __cplusplus */

#ifndef __cplusplus
typedef enum __x_ABI_CWindows_CMedia_CClosedCaptioning_CClosedCaptionSize __x_ABI_CWindows_CMedia_CClosedCaptioning_CClosedCaptionSize;
#endif /* __cplusplus */

#ifndef __cplusplus
typedef enum __x_ABI_CWindows_CMedia_CClosedCaptioning_CClosedCaptionStyle __x_ABI_CWindows_CMedia_CClosedCaptioning_CClosedCaptionStyle;
#endif /* __cplusplus */

#ifndef ____x_ABI_CWindows_CMedia_CClosedCaptioning_CIClosedCaptionPropertiesStatics_FWD_DEFINED__
#define ____x_ABI_CWindows_CMedia_CClosedCaptioning_CIClosedCaptionPropertiesStatics_FWD_DEFINED__
typedef interface __x_ABI_CWindows_CMedia_CClosedCaptioning_CIClosedCaptionPropertiesStatics __x_ABI_CWindows_CMedia_CClosedCaptioning_CIClosedCaptionPropertiesStatics;
#ifdef __cplusplus
#define __x_ABI_CWindows_CMedia_CClosedCaptioning_CIClosedCaptionPropertiesStatics ABI::Windows::Media::ClosedCaptioning::IClosedCaptionPropertiesStatics
namespace ABI {
    namespace Windows {
        namespace Media {
            namespace ClosedCaptioning {
                interface IClosedCaptionPropertiesStatics;
            }
        }
    }
}
#endif /* __cplusplus */
#endif

#if WINDOWS_FOUNDATION_UNIVERSALAPICONTRACT_VERSION >= 0x10000
#ifdef __cplusplus
} /* extern "C" */
namespace ABI {
    namespace Windows {
        namespace Media {
            namespace ClosedCaptioning {
                enum ClosedCaptionColor {
                    ClosedCaptionColor_Default = 0,
                    ClosedCaptionColor_White = 1,
                    ClosedCaptionColor_Black = 2,
                    ClosedCaptionColor_Red = 3,
                    ClosedCaptionColor_Green = 4,
                    ClosedCaptionColor_Blue = 5,
                    ClosedCaptionColor_Yellow = 6,
                    ClosedCaptionColor_Magenta = 7,
                    ClosedCaptionColor_Cyan = 8
                };
            }
        }
    }
}
extern "C" {
#else
enum __x_ABI_CWindows_CMedia_CClosedCaptioning_CClosedCaptionColor {
    ClosedCaptionColor_Default = 0,
    ClosedCaptionColor_White = 1,
    ClosedCaptionColor_Black = 2,
    ClosedCaptionColor_Red = 3,
    ClosedCaptionColor_Green = 4,
    ClosedCaptionColor_Blue = 5,
    ClosedCaptionColor_Yellow = 6,
    ClosedCaptionColor_Magenta = 7,
    ClosedCaptionColor_Cyan = 8
};
#ifdef WIDL_using_Windows_Media_ClosedCaptioning
#define ClosedCaptionColor __x_ABI_CWindows_CMedia_CClosedCaptioning_CClosedCaptionColor
#endif /* WIDL_using_Windows_Media_ClosedCaptioning */
#endif

#endif /* WINDOWS_FOUNDATION_UNIVERSALAPICONTRACT_VERSION >= 0x10000 */
#if WINDOWS_FOUNDATION_UNIVERSALAPICONTRACT_VERSION >= 0x10000
#ifdef __cplusplus
} /* extern "C" */
namespace ABI {
    namespace Windows {
        namespace Media {
            namespace ClosedCaptioning {
                enum ClosedCaptionEdgeEffect {
                    ClosedCaptionEdgeEffect_Default = 0,
                    ClosedCaptionEdgeEffect_None = 1,
                    ClosedCaptionEdgeEffect_Raised = 2,
                    ClosedCaptionEdgeEffect_Depressed = 3,
                    ClosedCaptionEdgeEffect_Uniform = 4,
                    ClosedCaptionEdgeEffect_DropShadow = 5
                };
            }
        }
    }
}
extern "C" {
#else
enum __x_ABI_CWindows_CMedia_CClosedCaptioning_CClosedCaptionEdgeEffect {
    ClosedCaptionEdgeEffect_Default = 0,
    ClosedCaptionEdgeEffect_None = 1,
    ClosedCaptionEdgeEffect_Raised = 2,
    ClosedCaptionEdgeEffect_Depressed = 3,
    ClosedCaptionEdgeEffect_Uniform = 4,
    ClosedCaptionEdgeEffect_DropShadow = 5
};
#ifdef WIDL_using_Windows_Media_ClosedCaptioning
#define ClosedCaptionEdgeEffect __x_ABI_CWindows_CMedia_CClosedCaptioning_CClosedCaptionEdgeEffect
#endif /* WIDL_using_Windows_Media_ClosedCaptioning */
#endif

#endif /* WINDOWS_FOUNDATION_UNIVERSALAPICONTRACT_VERSION >= 0x10000 */
#if WINDOWS_FOUNDATION_UNIVERSALAPICONTRACT_VERSION >= 0x10000
#ifdef __cplusplus
} /* extern "C" */
namespace ABI {
    namespace Windows {
        namespace Media {
            namespace ClosedCaptioning {
                enum ClosedCaptionOpacity {
                    ClosedCaptionOpacity_Default = 0,
                    ClosedCaptionOpacity_OneHundredPercent = 1,
                    ClosedCaptionOpacity_SeventyFivePercent = 2,
                    ClosedCaptionOpacity_TwentyFivePercent = 3,
                    ClosedCaptionOpacity_ZeroPercent = 4
                };
            }
        }
    }
}
extern "C" {
#else
enum __x_ABI_CWindows_CMedia_CClosedCaptioning_CClosedCaptionOpacity {
    ClosedCaptionOpacity_Default = 0,
    ClosedCaptionOpacity_OneHundredPercent = 1,
    ClosedCaptionOpacity_SeventyFivePercent = 2,
    ClosedCaptionOpacity_TwentyFivePercent = 3,
    ClosedCaptionOpacity_ZeroPercent = 4
};
#ifdef WIDL_using_Windows_Media_ClosedCaptioning
#define ClosedCaptionOpacity __x_ABI_CWindows_CMedia_CClosedCaptioning_CClosedCaptionOpacity
#endif /* WIDL_using_Windows_Media_ClosedCaptioning */
#endif

#endif /* WINDOWS_FOUNDATION_UNIVERSALAPICONTRACT_VERSION >= 0x10000 */
#if WINDOWS_FOUNDATION_UNIVERSALAPICONTRACT_VERSION >= 0x10000
#ifdef __cplusplus
} /* extern "C" */
namespace ABI {
    namespace Windows {
        namespace Media {
            namespace ClosedCaptioning {
                enum ClosedCaptionSize {
                    ClosedCaptionSize_Default = 0,
                    ClosedCaptionSize_FiftyPercent = 1,
                    ClosedCaptionSize_OneHundredPercent = 2,
                    ClosedCaptionSize_OneHundredFiftyPercent = 3,
                    ClosedCaptionSize_TwoHundredPercent = 4
                };
            }
        }
    }
}
extern "C" {
#else
enum __x_ABI_CWindows_CMedia_CClosedCaptioning_CClosedCaptionSize {
    ClosedCaptionSize_Default = 0,
    ClosedCaptionSize_FiftyPercent = 1,
    ClosedCaptionSize_OneHundredPercent = 2,
    ClosedCaptionSize_OneHundredFiftyPercent = 3,
    ClosedCaptionSize_TwoHundredPercent = 4
};
#ifdef WIDL_using_Windows_Media_ClosedCaptioning
#define ClosedCaptionSize __x_ABI_CWindows_CMedia_CClosedCaptioning_CClosedCaptionSize
#endif /* WIDL_using_Windows_Media_ClosedCaptioning */
#endif

#endif /* WINDOWS_FOUNDATION_UNIVERSALAPICONTRACT_VERSION >= 0x10000 */
#if WINDOWS_FOUNDATION_UNIVERSALAPICONTRACT_VERSION >= 0x10000
#ifdef __cplusplus
} /* extern "C" */
namespace ABI {
    namespace Windows {
        namespace Media {
            namespace ClosedCaptioning {
                enum ClosedCaptionStyle {
                    ClosedCaptionStyle_Default = 0,
                    ClosedCaptionStyle_MonospacedWithSerifs = 1,
                    ClosedCaptionStyle_ProportionalWithSerifs = 2,
                    ClosedCaptionStyle_MonospacedWithoutSerifs = 3,
                    ClosedCaptionStyle_ProportionalWithoutSerifs = 4,
                    ClosedCaptionStyle_Casual = 5,
                    ClosedCaptionStyle_Cursive = 6,
                    ClosedCaptionStyle_SmallCapitals = 7
                };
            }
        }
    }
}
extern "C" {
#else
enum __x_ABI_CWindows_CMedia_CClosedCaptioning_CClosedCaptionStyle {
    ClosedCaptionStyle_Default = 0,
    ClosedCaptionStyle_MonospacedWithSerifs = 1,
    ClosedCaptionStyle_ProportionalWithSerifs = 2,
    ClosedCaptionStyle_MonospacedWithoutSerifs = 3,
    ClosedCaptionStyle_ProportionalWithoutSerifs = 4,
    ClosedCaptionStyle_Casual = 5,
    ClosedCaptionStyle_Cursive = 6,
    ClosedCaptionStyle_SmallCapitals = 7
};
#ifdef WIDL_using_Windows_Media_ClosedCaptioning
#define ClosedCaptionStyle __x_ABI_CWindows_CMedia_CClosedCaptioning_CClosedCaptionStyle
#endif /* WIDL_using_Windows_Media_ClosedCaptioning */
#endif

#endif /* WINDOWS_FOUNDATION_UNIVERSALAPICONTRACT_VERSION >= 0x10000 */
/*****************************************************************************
 * IClosedCaptionPropertiesStatics interface
 */
#if WINDOWS_FOUNDATION_UNIVERSALAPICONTRACT_VERSION >= 0x10000
#ifndef ____x_ABI_CWindows_CMedia_CClosedCaptioning_CIClosedCaptionPropertiesStatics_INTERFACE_DEFINED__
#define ____x_ABI_CWindows_CMedia_CClosedCaptioning_CIClosedCaptionPropertiesStatics_INTERFACE_DEFINED__

DEFINE_GUID(IID___x_ABI_CWindows_CMedia_CClosedCaptioning_CIClosedCaptionPropertiesStatics, 0x10aa1f84, 0xcc30, 0x4141, 0xb5,0x03, 0x52,0x72,0x28,0x9e,0x0c,0x20);
#if defined(__cplusplus) && !defined(CINTERFACE)
} /* extern "C" */
namespace ABI {
    namespace Windows {
        namespace Media {
            namespace ClosedCaptioning {
                MIDL_INTERFACE("10aa1f84-cc30-4141-b503-5272289e0c20")
                IClosedCaptionPropertiesStatics : public IInspectable
                {
                    virtual HRESULT STDMETHODCALLTYPE get_FontColor(
                        enum ClosedCaptionColor *value) = 0;

                    virtual HRESULT STDMETHODCALLTYPE get_ComputedFontColor(
                        struct Color *value) = 0;

                    virtual HRESULT STDMETHODCALLTYPE get_FontOpacity(
                        enum ClosedCaptionOpacity *value) = 0;

                    virtual HRESULT STDMETHODCALLTYPE get_FontSize(
                        enum ClosedCaptionSize *value) = 0;

                    virtual HRESULT STDMETHODCALLTYPE get_FontStyle(
                        enum ClosedCaptionStyle *value) = 0;

                    virtual HRESULT STDMETHODCALLTYPE get_FontEffect(
                        enum ClosedCaptionEdgeEffect *value) = 0;

                    virtual HRESULT STDMETHODCALLTYPE get_BackgroundColor(
                        enum ClosedCaptionColor *value) = 0;

                    virtual HRESULT STDMETHODCALLTYPE get_ComputedBackgroundColor(
                        struct Color *value) = 0;

                    virtual HRESULT STDMETHODCALLTYPE get_BackgroundOpacity(
                        enum ClosedCaptionOpacity *value) = 0;

                    virtual HRESULT STDMETHODCALLTYPE get_RegionColor(
                        enum ClosedCaptionColor *value) = 0;

                    virtual HRESULT STDMETHODCALLTYPE get_ComputedRegionColor(
                        struct Color *value) = 0;

                    virtual HRESULT STDMETHODCALLTYPE get_RegionOpacity(
                        enum ClosedCaptionOpacity *value) = 0;

                };
            }
        }
    }
}
extern "C" {
#ifdef __CRT_UUID_DECL
__CRT_UUID_DECL(__x_ABI_CWindows_CMedia_CClosedCaptioning_CIClosedCaptionPropertiesStatics, 0x10aa1f84, 0xcc30, 0x4141, 0xb5,0x03, 0x52,0x72,0x28,0x9e,0x0c,0x20)
#endif
#else
typedef struct __x_ABI_CWindows_CMedia_CClosedCaptioning_CIClosedCaptionPropertiesStaticsVtbl {
    BEGIN_INTERFACE

    /*** IUnknown methods ***/
    HRESULT (STDMETHODCALLTYPE *QueryInterface)(
        __x_ABI_CWindows_CMedia_CClosedCaptioning_CIClosedCaptionPropertiesStatics *This,
        REFIID riid,
        void **ppvObject);

    ULONG (STDMETHODCALLTYPE *AddRef)(
        __x_ABI_CWindows_CMedia_CClosedCaptioning_CIClosedCaptionPropertiesStatics *This);

    ULONG (STDMETHODCALLTYPE *Release)(
        __x_ABI_CWindows_CMedia_CClosedCaptioning_CIClosedCaptionPropertiesStatics *This);

    /*** IInspectable methods ***/
    HRESULT (STDMETHODCALLTYPE *GetIids)(
        __x_ABI_CWindows_CMedia_CClosedCaptioning_CIClosedCaptionPropertiesStatics *This,
        ULONG *iidCount,
        IID **iids);

    HRESULT (STDMETHODCALLTYPE *GetRuntimeClassName)(
        __x_ABI_CWindows_CMedia_CClosedCaptioning_CIClosedCaptionPropertiesStatics *This,
        HSTRING *className);

    HRESULT (STDMETHODCALLTYPE *GetTrustLevel)(
        __x_ABI_CWindows_CMedia_CClosedCaptioning_CIClosedCaptionPropertiesStatics *This,
        TrustLevel *trustLevel);

    /*** IClosedCaptionPropertiesStatics methods ***/
    HRESULT (STDMETHODCALLTYPE *get_FontColor)(
        __x_ABI_CWindows_CMedia_CClosedCaptioning_CIClosedCaptionPropertiesStatics *This,
        enum __x_ABI_CWindows_CMedia_CClosedCaptioning_CClosedCaptionColor *value);

    HRESULT (STDMETHODCALLTYPE *get_ComputedFontColor)(
        __x_ABI_CWindows_CMedia_CClosedCaptioning_CIClosedCaptionPropertiesStatics *This,
        struct __x_ABI_CWindows_CUI_CColor *value);

    HRESULT (STDMETHODCALLTYPE *get_FontOpacity)(
        __x_ABI_CWindows_CMedia_CClosedCaptioning_CIClosedCaptionPropertiesStatics *This,
        enum __x_ABI_CWindows_CMedia_CClosedCaptioning_CClosedCaptionOpacity *value);

    HRESULT (STDMETHODCALLTYPE *get_FontSize)(
        __x_ABI_CWindows_CMedia_CClosedCaptioning_CIClosedCaptionPropertiesStatics *This,
        enum __x_ABI_CWindows_CMedia_CClosedCaptioning_CClosedCaptionSize *value);

    HRESULT (STDMETHODCALLTYPE *get_FontStyle)(
        __x_ABI_CWindows_CMedia_CClosedCaptioning_CIClosedCaptionPropertiesStatics *This,
        enum __x_ABI_CWindows_CMedia_CClosedCaptioning_CClosedCaptionStyle *value);

    HRESULT (STDMETHODCALLTYPE *get_FontEffect)(
        __x_ABI_CWindows_CMedia_CClosedCaptioning_CIClosedCaptionPropertiesStatics *This,
        enum __x_ABI_CWindows_CMedia_CClosedCaptioning_CClosedCaptionEdgeEffect *value);

    HRESULT (STDMETHODCALLTYPE *get_BackgroundColor)(
        __x_ABI_CWindows_CMedia_CClosedCaptioning_CIClosedCaptionPropertiesStatics *This,
        enum __x_ABI_CWindows_CMedia_CClosedCaptioning_CClosedCaptionColor *value);

    HRESULT (STDMETHODCALLTYPE *get_ComputedBackgroundColor)(
        __x_ABI_CWindows_CMedia_CClosedCaptioning_CIClosedCaptionPropertiesStatics *This,
        struct __x_ABI_CWindows_CUI_CColor *value);

    HRESULT (STDMETHODCALLTYPE *get_BackgroundOpacity)(
        __x_ABI_CWindows_CMedia_CClosedCaptioning_CIClosedCaptionPropertiesStatics *This,
        enum __x_ABI_CWindows_CMedia_CClosedCaptioning_CClosedCaptionOpacity *value);

    HRESULT (STDMETHODCALLTYPE *get_RegionColor)(
        __x_ABI_CWindows_CMedia_CClosedCaptioning_CIClosedCaptionPropertiesStatics *This,
        enum __x_ABI_CWindows_CMedia_CClosedCaptioning_CClosedCaptionColor *value);

    HRESULT (STDMETHODCALLTYPE *get_ComputedRegionColor)(
        __x_ABI_CWindows_CMedia_CClosedCaptioning_CIClosedCaptionPropertiesStatics *This,
        struct __x_ABI_CWindows_CUI_CColor *value);

    HRESULT (STDMETHODCALLTYPE *get_RegionOpacity)(
        __x_ABI_CWindows_CMedia_CClosedCaptioning_CIClosedCaptionPropertiesStatics *This,
        enum __x_ABI_CWindows_CMedia_CClosedCaptioning_CClosedCaptionOpacity *value);

    END_INTERFACE
} __x_ABI_CWindows_CMedia_CClosedCaptioning_CIClosedCaptionPropertiesStaticsVtbl;

interface __x_ABI_CWindows_CMedia_CClosedCaptioning_CIClosedCaptionPropertiesStatics {
    CONST_VTBL __x_ABI_CWindows_CMedia_CClosedCaptioning_CIClosedCaptionPropertiesStaticsVtbl* lpVtbl;
};

#ifdef COBJMACROS
#ifndef WIDL_C_INLINE_WRAPPERS
/*** IUnknown methods ***/
#define __x_ABI_CWindows_CMedia_CClosedCaptioning_CIClosedCaptionPropertiesStatics_QueryInterface(This,riid,ppvObject) (This)->lpVtbl->QueryInterface(This,riid,ppvObject)
#define __x_ABI_CWindows_CMedia_CClosedCaptioning_CIClosedCaptionPropertiesStatics_AddRef(This) (This)->lpVtbl->AddRef(This)
#define __x_ABI_CWindows_CMedia_CClosedCaptioning_CIClosedCaptionPropertiesStatics_Release(This) (This)->lpVtbl->Release(This)
/*** IInspectable methods ***/
#define __x_ABI_CWindows_CMedia_CClosedCaptioning_CIClosedCaptionPropertiesStatics_GetIids(This,iidCount,iids) (This)->lpVtbl->GetIids(This,iidCount,iids)
#define __x_ABI_CWindows_CMedia_CClosedCaptioning_CIClosedCaptionPropertiesStatics_GetRuntimeClassName(This,className) (This)->lpVtbl->GetRuntimeClassName(This,className)
#define __x_ABI_CWindows_CMedia_CClosedCaptioning_CIClosedCaptionPropertiesStatics_GetTrustLevel(This,trustLevel) (This)->lpVtbl->GetTrustLevel(This,trustLevel)
/*** IClosedCaptionPropertiesStatics methods ***/
#define __x_ABI_CWindows_CMedia_CClosedCaptioning_CIClosedCaptionPropertiesStatics_get_FontColor(This,value) (This)->lpVtbl->get_FontColor(This,value)
#define __x_ABI_CWindows_CMedia_CClosedCaptioning_CIClosedCaptionPropertiesStatics_get_ComputedFontColor(This,value) (This)->lpVtbl->get_ComputedFontColor(This,value)
#define __x_ABI_CWindows_CMedia_CClosedCaptioning_CIClosedCaptionPropertiesStatics_get_FontOpacity(This,value) (This)->lpVtbl->get_FontOpacity(This,value)
#define __x_ABI_CWindows_CMedia_CClosedCaptioning_CIClosedCaptionPropertiesStatics_get_FontSize(This,value) (This)->lpVtbl->get_FontSize(This,value)
#define __x_ABI_CWindows_CMedia_CClosedCaptioning_CIClosedCaptionPropertiesStatics_get_FontStyle(This,value) (This)->lpVtbl->get_FontStyle(This,value)
#define __x_ABI_CWindows_CMedia_CClosedCaptioning_CIClosedCaptionPropertiesStatics_get_FontEffect(This,value) (This)->lpVtbl->get_FontEffect(This,value)
#define __x_ABI_CWindows_CMedia_CClosedCaptioning_CIClosedCaptionPropertiesStatics_get_BackgroundColor(This,value) (This)->lpVtbl->get_BackgroundColor(This,value)
#define __x_ABI_CWindows_CMedia_CClosedCaptioning_CIClosedCaptionPropertiesStatics_get_ComputedBackgroundColor(This,value) (This)->lpVtbl->get_ComputedBackgroundColor(This,value)
#define __x_ABI_CWindows_CMedia_CClosedCaptioning_CIClosedCaptionPropertiesStatics_get_BackgroundOpacity(This,value) (This)->lpVtbl->get_BackgroundOpacity(This,value)
#define __x_ABI_CWindows_CMedia_CClosedCaptioning_CIClosedCaptionPropertiesStatics_get_RegionColor(This,value) (This)->lpVtbl->get_RegionColor(This,value)
#define __x_ABI_CWindows_CMedia_CClosedCaptioning_CIClosedCaptionPropertiesStatics_get_ComputedRegionColor(This,value) (This)->lpVtbl->get_ComputedRegionColor(This,value)
#define __x_ABI_CWindows_CMedia_CClosedCaptioning_CIClosedCaptionPropertiesStatics_get_RegionOpacity(This,value) (This)->lpVtbl->get_RegionOpacity(This,value)
#else
/*** IUnknown methods ***/
static __WIDL_INLINE HRESULT __x_ABI_CWindows_CMedia_CClosedCaptioning_CIClosedCaptionPropertiesStatics_QueryInterface(__x_ABI_CWindows_CMedia_CClosedCaptioning_CIClosedCaptionPropertiesStatics* This,REFIID riid,void **ppvObject) {
    return This->lpVtbl->QueryInterface(This,riid,ppvObject);
}
static __WIDL_INLINE ULONG __x_ABI_CWindows_CMedia_CClosedCaptioning_CIClosedCaptionPropertiesStatics_AddRef(__x_ABI_CWindows_CMedia_CClosedCaptioning_CIClosedCaptionPropertiesStatics* This) {
    return This->lpVtbl->AddRef(This);
}
static __WIDL_INLINE ULONG __x_ABI_CWindows_CMedia_CClosedCaptioning_CIClosedCaptionPropertiesStatics_Release(__x_ABI_CWindows_CMedia_CClosedCaptioning_CIClosedCaptionPropertiesStatics* This) {
    return This->lpVtbl->Release(This);
}
/*** IInspectable methods ***/
static __WIDL_INLINE HRESULT __x_ABI_CWindows_CMedia_CClosedCaptioning_CIClosedCaptionPropertiesStatics_GetIids(__x_ABI_CWindows_CMedia_CClosedCaptioning_CIClosedCaptionPropertiesStatics* This,ULONG *iidCount,IID **iids) {
    return This->lpVtbl->GetIids(This,iidCount,iids);
}
static __WIDL_INLINE HRESULT __x_ABI_CWindows_CMedia_CClosedCaptioning_CIClosedCaptionPropertiesStatics_GetRuntimeClassName(__x_ABI_CWindows_CMedia_CClosedCaptioning_CIClosedCaptionPropertiesStatics* This,HSTRING *className) {
    return This->lpVtbl->GetRuntimeClassName(This,className);
}
static __WIDL_INLINE HRESULT __x_ABI_CWindows_CMedia_CClosedCaptioning_CIClosedCaptionPropertiesStatics_GetTrustLevel(__x_ABI_CWindows_CMedia_CClosedCaptioning_CIClosedCaptionPropertiesStatics* This,TrustLevel *trustLevel) {
    return This->lpVtbl->GetTrustLevel(This,trustLevel);
}
/*** IClosedCaptionPropertiesStatics methods ***/
static __WIDL_INLINE HRESULT __x_ABI_CWindows_CMedia_CClosedCaptioning_CIClosedCaptionPropertiesStatics_get_FontColor(__x_ABI_CWindows_CMedia_CClosedCaptioning_CIClosedCaptionPropertiesStatics* This,enum __x_ABI_CWindows_CMedia_CClosedCaptioning_CClosedCaptionColor *value) {
    return This->lpVtbl->get_FontColor(This,value);
}
static __WIDL_INLINE HRESULT __x_ABI_CWindows_CMedia_CClosedCaptioning_CIClosedCaptionPropertiesStatics_get_ComputedFontColor(__x_ABI_CWindows_CMedia_CClosedCaptioning_CIClosedCaptionPropertiesStatics* This,struct __x_ABI_CWindows_CUI_CColor *value) {
    return This->lpVtbl->get_ComputedFontColor(This,value);
}
static __WIDL_INLINE HRESULT __x_ABI_CWindows_CMedia_CClosedCaptioning_CIClosedCaptionPropertiesStatics_get_FontOpacity(__x_ABI_CWindows_CMedia_CClosedCaptioning_CIClosedCaptionPropertiesStatics* This,enum __x_ABI_CWindows_CMedia_CClosedCaptioning_CClosedCaptionOpacity *value) {
    return This->lpVtbl->get_FontOpacity(This,value);
}
static __WIDL_INLINE HRESULT __x_ABI_CWindows_CMedia_CClosedCaptioning_CIClosedCaptionPropertiesStatics_get_FontSize(__x_ABI_CWindows_CMedia_CClosedCaptioning_CIClosedCaptionPropertiesStatics* This,enum __x_ABI_CWindows_CMedia_CClosedCaptioning_CClosedCaptionSize *value) {
    return This->lpVtbl->get_FontSize(This,value);
}
static __WIDL_INLINE HRESULT __x_ABI_CWindows_CMedia_CClosedCaptioning_CIClosedCaptionPropertiesStatics_get_FontStyle(__x_ABI_CWindows_CMedia_CClosedCaptioning_CIClosedCaptionPropertiesStatics* This,enum __x_ABI_CWindows_CMedia_CClosedCaptioning_CClosedCaptionStyle *value) {
    return This->lpVtbl->get_FontStyle(This,value);
}
static __WIDL_INLINE HRESULT __x_ABI_CWindows_CMedia_CClosedCaptioning_CIClosedCaptionPropertiesStatics_get_FontEffect(__x_ABI_CWindows_CMedia_CClosedCaptioning_CIClosedCaptionPropertiesStatics* This,enum __x_ABI_CWindows_CMedia_CClosedCaptioning_CClosedCaptionEdgeEffect *value) {
    return This->lpVtbl->get_FontEffect(This,value);
}
static __WIDL_INLINE HRESULT __x_ABI_CWindows_CMedia_CClosedCaptioning_CIClosedCaptionPropertiesStatics_get_BackgroundColor(__x_ABI_CWindows_CMedia_CClosedCaptioning_CIClosedCaptionPropertiesStatics* This,enum __x_ABI_CWindows_CMedia_CClosedCaptioning_CClosedCaptionColor *value) {
    return This->lpVtbl->get_BackgroundColor(This,value);
}
static __WIDL_INLINE HRESULT __x_ABI_CWindows_CMedia_CClosedCaptioning_CIClosedCaptionPropertiesStatics_get_ComputedBackgroundColor(__x_ABI_CWindows_CMedia_CClosedCaptioning_CIClosedCaptionPropertiesStatics* This,struct __x_ABI_CWindows_CUI_CColor *value) {
    return This->lpVtbl->get_ComputedBackgroundColor(This,value);
}
static __WIDL_INLINE HRESULT __x_ABI_CWindows_CMedia_CClosedCaptioning_CIClosedCaptionPropertiesStatics_get_BackgroundOpacity(__x_ABI_CWindows_CMedia_CClosedCaptioning_CIClosedCaptionPropertiesStatics* This,enum __x_ABI_CWindows_CMedia_CClosedCaptioning_CClosedCaptionOpacity *value) {
    return This->lpVtbl->get_BackgroundOpacity(This,value);
}
static __WIDL_INLINE HRESULT __x_ABI_CWindows_CMedia_CClosedCaptioning_CIClosedCaptionPropertiesStatics_get_RegionColor(__x_ABI_CWindows_CMedia_CClosedCaptioning_CIClosedCaptionPropertiesStatics* This,enum __x_ABI_CWindows_CMedia_CClosedCaptioning_CClosedCaptionColor *value) {
    return This->lpVtbl->get_RegionColor(This,value);
}
static __WIDL_INLINE HRESULT __x_ABI_CWindows_CMedia_CClosedCaptioning_CIClosedCaptionPropertiesStatics_get_ComputedRegionColor(__x_ABI_CWindows_CMedia_CClosedCaptioning_CIClosedCaptionPropertiesStatics* This,struct __x_ABI_CWindows_CUI_CColor *value) {
    return This->lpVtbl->get_ComputedRegionColor(This,value);
}
static __WIDL_INLINE HRESULT __x_ABI_CWindows_CMedia_CClosedCaptioning_CIClosedCaptionPropertiesStatics_get_RegionOpacity(__x_ABI_CWindows_CMedia_CClosedCaptioning_CIClosedCaptionPropertiesStatics* This,enum __x_ABI_CWindows_CMedia_CClosedCaptioning_CClosedCaptionOpacity *value) {
    return This->lpVtbl->get_RegionOpacity(This,value);
}
#endif
#ifdef WIDL_using_Windows_Media_ClosedCaptioning
#define IID_IClosedCaptionPropertiesStatics IID___x_ABI_CWindows_CMedia_CClosedCaptioning_CIClosedCaptionPropertiesStatics
#define IClosedCaptionPropertiesStaticsVtbl __x_ABI_CWindows_CMedia_CClosedCaptioning_CIClosedCaptionPropertiesStaticsVtbl
#define IClosedCaptionPropertiesStatics __x_ABI_CWindows_CMedia_CClosedCaptioning_CIClosedCaptionPropertiesStatics
#define IClosedCaptionPropertiesStatics_QueryInterface __x_ABI_CWindows_CMedia_CClosedCaptioning_CIClosedCaptionPropertiesStatics_QueryInterface
#define IClosedCaptionPropertiesStatics_AddRef __x_ABI_CWindows_CMedia_CClosedCaptioning_CIClosedCaptionPropertiesStatics_AddRef
#define IClosedCaptionPropertiesStatics_Release __x_ABI_CWindows_CMedia_CClosedCaptioning_CIClosedCaptionPropertiesStatics_Release
#define IClosedCaptionPropertiesStatics_GetIids __x_ABI_CWindows_CMedia_CClosedCaptioning_CIClosedCaptionPropertiesStatics_GetIids
#define IClosedCaptionPropertiesStatics_GetRuntimeClassName __x_ABI_CWindows_CMedia_CClosedCaptioning_CIClosedCaptionPropertiesStatics_GetRuntimeClassName
#define IClosedCaptionPropertiesStatics_GetTrustLevel __x_ABI_CWindows_CMedia_CClosedCaptioning_CIClosedCaptionPropertiesStatics_GetTrustLevel
#define IClosedCaptionPropertiesStatics_get_FontColor __x_ABI_CWindows_CMedia_CClosedCaptioning_CIClosedCaptionPropertiesStatics_get_FontColor
#define IClosedCaptionPropertiesStatics_get_ComputedFontColor __x_ABI_CWindows_CMedia_CClosedCaptioning_CIClosedCaptionPropertiesStatics_get_ComputedFontColor
#define IClosedCaptionPropertiesStatics_get_FontOpacity __x_ABI_CWindows_CMedia_CClosedCaptioning_CIClosedCaptionPropertiesStatics_get_FontOpacity
#define IClosedCaptionPropertiesStatics_get_FontSize __x_ABI_CWindows_CMedia_CClosedCaptioning_CIClosedCaptionPropertiesStatics_get_FontSize
#define IClosedCaptionPropertiesStatics_get_FontStyle __x_ABI_CWindows_CMedia_CClosedCaptioning_CIClosedCaptionPropertiesStatics_get_FontStyle
#define IClosedCaptionPropertiesStatics_get_FontEffect __x_ABI_CWindows_CMedia_CClosedCaptioning_CIClosedCaptionPropertiesStatics_get_FontEffect
#define IClosedCaptionPropertiesStatics_get_BackgroundColor __x_ABI_CWindows_CMedia_CClosedCaptioning_CIClosedCaptionPropertiesStatics_get_BackgroundColor
#define IClosedCaptionPropertiesStatics_get_ComputedBackgroundColor __x_ABI_CWindows_CMedia_CClosedCaptioning_CIClosedCaptionPropertiesStatics_get_ComputedBackgroundColor
#define IClosedCaptionPropertiesStatics_get_BackgroundOpacity __x_ABI_CWindows_CMedia_CClosedCaptioning_CIClosedCaptionPropertiesStatics_get_BackgroundOpacity
#define IClosedCaptionPropertiesStatics_get_RegionColor __x_ABI_CWindows_CMedia_CClosedCaptioning_CIClosedCaptionPropertiesStatics_get_RegionColor
#define IClosedCaptionPropertiesStatics_get_ComputedRegionColor __x_ABI_CWindows_CMedia_CClosedCaptioning_CIClosedCaptionPropertiesStatics_get_ComputedRegionColor
#define IClosedCaptionPropertiesStatics_get_RegionOpacity __x_ABI_CWindows_CMedia_CClosedCaptioning_CIClosedCaptionPropertiesStatics_get_RegionOpacity
#endif /* WIDL_using_Windows_Media_ClosedCaptioning */
#endif

#endif

#endif  /* ____x_ABI_CWindows_CMedia_CClosedCaptioning_CIClosedCaptionPropertiesStatics_INTERFACE_DEFINED__ */
#endif /* WINDOWS_FOUNDATION_UNIVERSALAPICONTRACT_VERSION >= 0x10000 */

/*
 * Class Windows.Media.ClosedCaptioning.ClosedCaptionProperties
 */
#if WINDOWS_FOUNDATION_UNIVERSALAPICONTRACT_VERSION >= 0x10000
#ifndef RUNTIMECLASS_Windows_Media_ClosedCaptioning_ClosedCaptionProperties_DEFINED
#define RUNTIMECLASS_Windows_Media_ClosedCaptioning_ClosedCaptionProperties_DEFINED
#if !defined(_MSC_VER) && !defined(__MINGW32__)
static const WCHAR RuntimeClass_Windows_Media_ClosedCaptioning_ClosedCaptionProperties[] = {'W','i','n','d','o','w','s','.','M','e','d','i','a','.','C','l','o','s','e','d','C','a','p','t','i','o','n','i','n','g','.','C','l','o','s','e','d','C','a','p','t','i','o','n','P','r','o','p','e','r','t','i','e','s',0};
#elif defined(__GNUC__) && !defined(__cplusplus)
const DECLSPEC_SELECTANY WCHAR RuntimeClass_Windows_Media_ClosedCaptioning_ClosedCaptionProperties[] = L"Windows.Media.ClosedCaptioning.ClosedCaptionProperties";
#else
extern const DECLSPEC_SELECTANY WCHAR RuntimeClass_Windows_Media_ClosedCaptioning_ClosedCaptionProperties[] = {'W','i','n','d','o','w','s','.','M','e','d','i','a','.','C','l','o','s','e','d','C','a','p','t','i','o','n','i','n','g','.','C','l','o','s','e','d','C','a','p','t','i','o','n','P','r','o','p','e','r','t','i','e','s',0};
#endif
#endif /* RUNTIMECLASS_Windows_Media_ClosedCaptioning_ClosedCaptionProperties_DEFINED */
#endif /* WINDOWS_FOUNDATION_UNIVERSALAPICONTRACT_VERSION >= 0x10000 */

/* Begin additional prototypes for all interfaces */


/* End additional prototypes */

#ifdef __cplusplus
}
#endif

#endif /* __windows_media_closedcaptioning_h__ */
