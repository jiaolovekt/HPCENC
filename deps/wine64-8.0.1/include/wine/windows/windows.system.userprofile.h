/*** Autogenerated by WIDL 8.0.1 from ../include/windows.system.userprofile.idl - Do not edit ***/

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

#ifndef __windows_system_userprofile_h__
#define __windows_system_userprofile_h__

#ifndef __WIDL_INLINE
#if defined(__cplusplus) || defined(_MSC_VER)
#define __WIDL_INLINE inline
#elif defined(__GNUC__)
#define __WIDL_INLINE __inline__
#endif
#endif

/* Forward declarations */

#ifndef ____x_ABI_CWindows_CSystem_CUserProfile_CIGlobalizationPreferencesStatics_FWD_DEFINED__
#define ____x_ABI_CWindows_CSystem_CUserProfile_CIGlobalizationPreferencesStatics_FWD_DEFINED__
typedef interface __x_ABI_CWindows_CSystem_CUserProfile_CIGlobalizationPreferencesStatics __x_ABI_CWindows_CSystem_CUserProfile_CIGlobalizationPreferencesStatics;
#ifdef __cplusplus
#define __x_ABI_CWindows_CSystem_CUserProfile_CIGlobalizationPreferencesStatics ABI::Windows::System::UserProfile::IGlobalizationPreferencesStatics
namespace ABI {
    namespace Windows {
        namespace System {
            namespace UserProfile {
                interface IGlobalizationPreferencesStatics;
            }
        }
    }
}
#endif /* __cplusplus */
#endif

#ifndef ____x_ABI_CWindows_CSystem_CUserProfile_CGlobalizationPreferences_FWD_DEFINED__
#define ____x_ABI_CWindows_CSystem_CUserProfile_CGlobalizationPreferences_FWD_DEFINED__
#ifdef __cplusplus
namespace ABI {
    namespace Windows {
        namespace System {
            namespace UserProfile {
                class GlobalizationPreferences;
            }
        }
    }
}
#else
typedef struct __x_ABI_CWindows_CSystem_CUserProfile_CGlobalizationPreferences __x_ABI_CWindows_CSystem_CUserProfile_CGlobalizationPreferences;
#endif /* defined __cplusplus */
#endif /* defined ____x_ABI_CWindows_CSystem_CUserProfile_CGlobalizationPreferences_FWD_DEFINED__ */

/* Headers for imported files */

#include <inspectable.h>
#include <asyncinfo.h>
#include <eventtoken.h>
#include <windowscontracts.h>
#include <windows.foundation.h>
#include <windows.globalization.h>

#ifdef __cplusplus
extern "C" {
#endif

#ifndef ____x_ABI_CWindows_CSystem_CUserProfile_CIGlobalizationPreferencesStatics_FWD_DEFINED__
#define ____x_ABI_CWindows_CSystem_CUserProfile_CIGlobalizationPreferencesStatics_FWD_DEFINED__
typedef interface __x_ABI_CWindows_CSystem_CUserProfile_CIGlobalizationPreferencesStatics __x_ABI_CWindows_CSystem_CUserProfile_CIGlobalizationPreferencesStatics;
#ifdef __cplusplus
#define __x_ABI_CWindows_CSystem_CUserProfile_CIGlobalizationPreferencesStatics ABI::Windows::System::UserProfile::IGlobalizationPreferencesStatics
namespace ABI {
    namespace Windows {
        namespace System {
            namespace UserProfile {
                interface IGlobalizationPreferencesStatics;
            }
        }
    }
}
#endif /* __cplusplus */
#endif

#ifndef ____x_ABI_CWindows_CSystem_CUserProfile_CIGlobalizationPreferencesStatics2_FWD_DEFINED__
#define ____x_ABI_CWindows_CSystem_CUserProfile_CIGlobalizationPreferencesStatics2_FWD_DEFINED__
typedef interface __x_ABI_CWindows_CSystem_CUserProfile_CIGlobalizationPreferencesStatics2 __x_ABI_CWindows_CSystem_CUserProfile_CIGlobalizationPreferencesStatics2;
#ifdef __cplusplus
#define __x_ABI_CWindows_CSystem_CUserProfile_CIGlobalizationPreferencesStatics2 ABI::Windows::System::UserProfile::IGlobalizationPreferencesStatics2
namespace ABI {
    namespace Windows {
        namespace System {
            namespace UserProfile {
                interface IGlobalizationPreferencesStatics2;
            }
        }
    }
}
#endif /* __cplusplus */
#endif

#ifndef ____x_ABI_CWindows_CSystem_CUserProfile_CIGlobalizationPreferencesStatics3_FWD_DEFINED__
#define ____x_ABI_CWindows_CSystem_CUserProfile_CIGlobalizationPreferencesStatics3_FWD_DEFINED__
typedef interface __x_ABI_CWindows_CSystem_CUserProfile_CIGlobalizationPreferencesStatics3 __x_ABI_CWindows_CSystem_CUserProfile_CIGlobalizationPreferencesStatics3;
#ifdef __cplusplus
#define __x_ABI_CWindows_CSystem_CUserProfile_CIGlobalizationPreferencesStatics3 ABI::Windows::System::UserProfile::IGlobalizationPreferencesStatics3
namespace ABI {
    namespace Windows {
        namespace System {
            namespace UserProfile {
                interface IGlobalizationPreferencesStatics3;
            }
        }
    }
}
#endif /* __cplusplus */
#endif

/*****************************************************************************
 * IGlobalizationPreferencesStatics interface
 */
#if WINDOWS_FOUNDATION_UNIVERSALAPICONTRACT_VERSION >= 0x10000
#ifndef ____x_ABI_CWindows_CSystem_CUserProfile_CIGlobalizationPreferencesStatics_INTERFACE_DEFINED__
#define ____x_ABI_CWindows_CSystem_CUserProfile_CIGlobalizationPreferencesStatics_INTERFACE_DEFINED__

DEFINE_GUID(IID___x_ABI_CWindows_CSystem_CUserProfile_CIGlobalizationPreferencesStatics, 0x01bf4326, 0xed37, 0x4e96, 0xb0,0xe9, 0xc1,0x34,0x0d,0x1e,0xa1,0x58);
#if defined(__cplusplus) && !defined(CINTERFACE)
} /* extern "C" */
namespace ABI {
    namespace Windows {
        namespace System {
            namespace UserProfile {
                MIDL_INTERFACE("01bf4326-ed37-4e96-b0e9-c1340d1ea158")
                IGlobalizationPreferencesStatics : public IInspectable
                {
                    virtual HRESULT STDMETHODCALLTYPE get_Calendars(
                        ABI::Windows::Foundation::Collections::IVectorView<HSTRING > **value) = 0;

                    virtual HRESULT STDMETHODCALLTYPE get_Clocks(
                        ABI::Windows::Foundation::Collections::IVectorView<HSTRING > **value) = 0;

                    virtual HRESULT STDMETHODCALLTYPE get_Currencies(
                        ABI::Windows::Foundation::Collections::IVectorView<HSTRING > **value) = 0;

                    virtual HRESULT STDMETHODCALLTYPE get_Languages(
                        ABI::Windows::Foundation::Collections::IVectorView<HSTRING > **value) = 0;

                    virtual HRESULT STDMETHODCALLTYPE get_HomeGeographicRegion(
                        HSTRING *value) = 0;

                    virtual HRESULT STDMETHODCALLTYPE get_WeekStartsOn(
                        enum DayOfWeek *value) = 0;

                };
            }
        }
    }
}
extern "C" {
#ifdef __CRT_UUID_DECL
__CRT_UUID_DECL(__x_ABI_CWindows_CSystem_CUserProfile_CIGlobalizationPreferencesStatics, 0x01bf4326, 0xed37, 0x4e96, 0xb0,0xe9, 0xc1,0x34,0x0d,0x1e,0xa1,0x58)
#endif
#else
typedef struct __x_ABI_CWindows_CSystem_CUserProfile_CIGlobalizationPreferencesStaticsVtbl {
    BEGIN_INTERFACE

    /*** IUnknown methods ***/
    HRESULT (STDMETHODCALLTYPE *QueryInterface)(
        __x_ABI_CWindows_CSystem_CUserProfile_CIGlobalizationPreferencesStatics *This,
        REFIID riid,
        void **ppvObject);

    ULONG (STDMETHODCALLTYPE *AddRef)(
        __x_ABI_CWindows_CSystem_CUserProfile_CIGlobalizationPreferencesStatics *This);

    ULONG (STDMETHODCALLTYPE *Release)(
        __x_ABI_CWindows_CSystem_CUserProfile_CIGlobalizationPreferencesStatics *This);

    /*** IInspectable methods ***/
    HRESULT (STDMETHODCALLTYPE *GetIids)(
        __x_ABI_CWindows_CSystem_CUserProfile_CIGlobalizationPreferencesStatics *This,
        ULONG *iidCount,
        IID **iids);

    HRESULT (STDMETHODCALLTYPE *GetRuntimeClassName)(
        __x_ABI_CWindows_CSystem_CUserProfile_CIGlobalizationPreferencesStatics *This,
        HSTRING *className);

    HRESULT (STDMETHODCALLTYPE *GetTrustLevel)(
        __x_ABI_CWindows_CSystem_CUserProfile_CIGlobalizationPreferencesStatics *This,
        TrustLevel *trustLevel);

    /*** IGlobalizationPreferencesStatics methods ***/
    HRESULT (STDMETHODCALLTYPE *get_Calendars)(
        __x_ABI_CWindows_CSystem_CUserProfile_CIGlobalizationPreferencesStatics *This,
        __FIVectorView_1_HSTRING **value);

    HRESULT (STDMETHODCALLTYPE *get_Clocks)(
        __x_ABI_CWindows_CSystem_CUserProfile_CIGlobalizationPreferencesStatics *This,
        __FIVectorView_1_HSTRING **value);

    HRESULT (STDMETHODCALLTYPE *get_Currencies)(
        __x_ABI_CWindows_CSystem_CUserProfile_CIGlobalizationPreferencesStatics *This,
        __FIVectorView_1_HSTRING **value);

    HRESULT (STDMETHODCALLTYPE *get_Languages)(
        __x_ABI_CWindows_CSystem_CUserProfile_CIGlobalizationPreferencesStatics *This,
        __FIVectorView_1_HSTRING **value);

    HRESULT (STDMETHODCALLTYPE *get_HomeGeographicRegion)(
        __x_ABI_CWindows_CSystem_CUserProfile_CIGlobalizationPreferencesStatics *This,
        HSTRING *value);

    HRESULT (STDMETHODCALLTYPE *get_WeekStartsOn)(
        __x_ABI_CWindows_CSystem_CUserProfile_CIGlobalizationPreferencesStatics *This,
        enum __x_ABI_CWindows_CGlobalization_CDayOfWeek *value);

    END_INTERFACE
} __x_ABI_CWindows_CSystem_CUserProfile_CIGlobalizationPreferencesStaticsVtbl;

interface __x_ABI_CWindows_CSystem_CUserProfile_CIGlobalizationPreferencesStatics {
    CONST_VTBL __x_ABI_CWindows_CSystem_CUserProfile_CIGlobalizationPreferencesStaticsVtbl* lpVtbl;
};

#ifdef COBJMACROS
#ifndef WIDL_C_INLINE_WRAPPERS
/*** IUnknown methods ***/
#define __x_ABI_CWindows_CSystem_CUserProfile_CIGlobalizationPreferencesStatics_QueryInterface(This,riid,ppvObject) (This)->lpVtbl->QueryInterface(This,riid,ppvObject)
#define __x_ABI_CWindows_CSystem_CUserProfile_CIGlobalizationPreferencesStatics_AddRef(This) (This)->lpVtbl->AddRef(This)
#define __x_ABI_CWindows_CSystem_CUserProfile_CIGlobalizationPreferencesStatics_Release(This) (This)->lpVtbl->Release(This)
/*** IInspectable methods ***/
#define __x_ABI_CWindows_CSystem_CUserProfile_CIGlobalizationPreferencesStatics_GetIids(This,iidCount,iids) (This)->lpVtbl->GetIids(This,iidCount,iids)
#define __x_ABI_CWindows_CSystem_CUserProfile_CIGlobalizationPreferencesStatics_GetRuntimeClassName(This,className) (This)->lpVtbl->GetRuntimeClassName(This,className)
#define __x_ABI_CWindows_CSystem_CUserProfile_CIGlobalizationPreferencesStatics_GetTrustLevel(This,trustLevel) (This)->lpVtbl->GetTrustLevel(This,trustLevel)
/*** IGlobalizationPreferencesStatics methods ***/
#define __x_ABI_CWindows_CSystem_CUserProfile_CIGlobalizationPreferencesStatics_get_Calendars(This,value) (This)->lpVtbl->get_Calendars(This,value)
#define __x_ABI_CWindows_CSystem_CUserProfile_CIGlobalizationPreferencesStatics_get_Clocks(This,value) (This)->lpVtbl->get_Clocks(This,value)
#define __x_ABI_CWindows_CSystem_CUserProfile_CIGlobalizationPreferencesStatics_get_Currencies(This,value) (This)->lpVtbl->get_Currencies(This,value)
#define __x_ABI_CWindows_CSystem_CUserProfile_CIGlobalizationPreferencesStatics_get_Languages(This,value) (This)->lpVtbl->get_Languages(This,value)
#define __x_ABI_CWindows_CSystem_CUserProfile_CIGlobalizationPreferencesStatics_get_HomeGeographicRegion(This,value) (This)->lpVtbl->get_HomeGeographicRegion(This,value)
#define __x_ABI_CWindows_CSystem_CUserProfile_CIGlobalizationPreferencesStatics_get_WeekStartsOn(This,value) (This)->lpVtbl->get_WeekStartsOn(This,value)
#else
/*** IUnknown methods ***/
static __WIDL_INLINE HRESULT __x_ABI_CWindows_CSystem_CUserProfile_CIGlobalizationPreferencesStatics_QueryInterface(__x_ABI_CWindows_CSystem_CUserProfile_CIGlobalizationPreferencesStatics* This,REFIID riid,void **ppvObject) {
    return This->lpVtbl->QueryInterface(This,riid,ppvObject);
}
static __WIDL_INLINE ULONG __x_ABI_CWindows_CSystem_CUserProfile_CIGlobalizationPreferencesStatics_AddRef(__x_ABI_CWindows_CSystem_CUserProfile_CIGlobalizationPreferencesStatics* This) {
    return This->lpVtbl->AddRef(This);
}
static __WIDL_INLINE ULONG __x_ABI_CWindows_CSystem_CUserProfile_CIGlobalizationPreferencesStatics_Release(__x_ABI_CWindows_CSystem_CUserProfile_CIGlobalizationPreferencesStatics* This) {
    return This->lpVtbl->Release(This);
}
/*** IInspectable methods ***/
static __WIDL_INLINE HRESULT __x_ABI_CWindows_CSystem_CUserProfile_CIGlobalizationPreferencesStatics_GetIids(__x_ABI_CWindows_CSystem_CUserProfile_CIGlobalizationPreferencesStatics* This,ULONG *iidCount,IID **iids) {
    return This->lpVtbl->GetIids(This,iidCount,iids);
}
static __WIDL_INLINE HRESULT __x_ABI_CWindows_CSystem_CUserProfile_CIGlobalizationPreferencesStatics_GetRuntimeClassName(__x_ABI_CWindows_CSystem_CUserProfile_CIGlobalizationPreferencesStatics* This,HSTRING *className) {
    return This->lpVtbl->GetRuntimeClassName(This,className);
}
static __WIDL_INLINE HRESULT __x_ABI_CWindows_CSystem_CUserProfile_CIGlobalizationPreferencesStatics_GetTrustLevel(__x_ABI_CWindows_CSystem_CUserProfile_CIGlobalizationPreferencesStatics* This,TrustLevel *trustLevel) {
    return This->lpVtbl->GetTrustLevel(This,trustLevel);
}
/*** IGlobalizationPreferencesStatics methods ***/
static __WIDL_INLINE HRESULT __x_ABI_CWindows_CSystem_CUserProfile_CIGlobalizationPreferencesStatics_get_Calendars(__x_ABI_CWindows_CSystem_CUserProfile_CIGlobalizationPreferencesStatics* This,__FIVectorView_1_HSTRING **value) {
    return This->lpVtbl->get_Calendars(This,value);
}
static __WIDL_INLINE HRESULT __x_ABI_CWindows_CSystem_CUserProfile_CIGlobalizationPreferencesStatics_get_Clocks(__x_ABI_CWindows_CSystem_CUserProfile_CIGlobalizationPreferencesStatics* This,__FIVectorView_1_HSTRING **value) {
    return This->lpVtbl->get_Clocks(This,value);
}
static __WIDL_INLINE HRESULT __x_ABI_CWindows_CSystem_CUserProfile_CIGlobalizationPreferencesStatics_get_Currencies(__x_ABI_CWindows_CSystem_CUserProfile_CIGlobalizationPreferencesStatics* This,__FIVectorView_1_HSTRING **value) {
    return This->lpVtbl->get_Currencies(This,value);
}
static __WIDL_INLINE HRESULT __x_ABI_CWindows_CSystem_CUserProfile_CIGlobalizationPreferencesStatics_get_Languages(__x_ABI_CWindows_CSystem_CUserProfile_CIGlobalizationPreferencesStatics* This,__FIVectorView_1_HSTRING **value) {
    return This->lpVtbl->get_Languages(This,value);
}
static __WIDL_INLINE HRESULT __x_ABI_CWindows_CSystem_CUserProfile_CIGlobalizationPreferencesStatics_get_HomeGeographicRegion(__x_ABI_CWindows_CSystem_CUserProfile_CIGlobalizationPreferencesStatics* This,HSTRING *value) {
    return This->lpVtbl->get_HomeGeographicRegion(This,value);
}
static __WIDL_INLINE HRESULT __x_ABI_CWindows_CSystem_CUserProfile_CIGlobalizationPreferencesStatics_get_WeekStartsOn(__x_ABI_CWindows_CSystem_CUserProfile_CIGlobalizationPreferencesStatics* This,enum __x_ABI_CWindows_CGlobalization_CDayOfWeek *value) {
    return This->lpVtbl->get_WeekStartsOn(This,value);
}
#endif
#ifdef WIDL_using_Windows_System_UserProfile
#define IID_IGlobalizationPreferencesStatics IID___x_ABI_CWindows_CSystem_CUserProfile_CIGlobalizationPreferencesStatics
#define IGlobalizationPreferencesStaticsVtbl __x_ABI_CWindows_CSystem_CUserProfile_CIGlobalizationPreferencesStaticsVtbl
#define IGlobalizationPreferencesStatics __x_ABI_CWindows_CSystem_CUserProfile_CIGlobalizationPreferencesStatics
#define IGlobalizationPreferencesStatics_QueryInterface __x_ABI_CWindows_CSystem_CUserProfile_CIGlobalizationPreferencesStatics_QueryInterface
#define IGlobalizationPreferencesStatics_AddRef __x_ABI_CWindows_CSystem_CUserProfile_CIGlobalizationPreferencesStatics_AddRef
#define IGlobalizationPreferencesStatics_Release __x_ABI_CWindows_CSystem_CUserProfile_CIGlobalizationPreferencesStatics_Release
#define IGlobalizationPreferencesStatics_GetIids __x_ABI_CWindows_CSystem_CUserProfile_CIGlobalizationPreferencesStatics_GetIids
#define IGlobalizationPreferencesStatics_GetRuntimeClassName __x_ABI_CWindows_CSystem_CUserProfile_CIGlobalizationPreferencesStatics_GetRuntimeClassName
#define IGlobalizationPreferencesStatics_GetTrustLevel __x_ABI_CWindows_CSystem_CUserProfile_CIGlobalizationPreferencesStatics_GetTrustLevel
#define IGlobalizationPreferencesStatics_get_Calendars __x_ABI_CWindows_CSystem_CUserProfile_CIGlobalizationPreferencesStatics_get_Calendars
#define IGlobalizationPreferencesStatics_get_Clocks __x_ABI_CWindows_CSystem_CUserProfile_CIGlobalizationPreferencesStatics_get_Clocks
#define IGlobalizationPreferencesStatics_get_Currencies __x_ABI_CWindows_CSystem_CUserProfile_CIGlobalizationPreferencesStatics_get_Currencies
#define IGlobalizationPreferencesStatics_get_Languages __x_ABI_CWindows_CSystem_CUserProfile_CIGlobalizationPreferencesStatics_get_Languages
#define IGlobalizationPreferencesStatics_get_HomeGeographicRegion __x_ABI_CWindows_CSystem_CUserProfile_CIGlobalizationPreferencesStatics_get_HomeGeographicRegion
#define IGlobalizationPreferencesStatics_get_WeekStartsOn __x_ABI_CWindows_CSystem_CUserProfile_CIGlobalizationPreferencesStatics_get_WeekStartsOn
#endif /* WIDL_using_Windows_System_UserProfile */
#endif

#endif

#endif  /* ____x_ABI_CWindows_CSystem_CUserProfile_CIGlobalizationPreferencesStatics_INTERFACE_DEFINED__ */
#endif /* WINDOWS_FOUNDATION_UNIVERSALAPICONTRACT_VERSION >= 0x10000 */

/*
 * Class Windows.System.UserProfile.GlobalizationPreferences
 */
#if WINDOWS_FOUNDATION_UNIVERSALAPICONTRACT_VERSION >= 0x10000
#ifndef RUNTIMECLASS_Windows_System_UserProfile_GlobalizationPreferences_DEFINED
#define RUNTIMECLASS_Windows_System_UserProfile_GlobalizationPreferences_DEFINED
#if !defined(_MSC_VER) && !defined(__MINGW32__)
static const WCHAR RuntimeClass_Windows_System_UserProfile_GlobalizationPreferences[] = {'W','i','n','d','o','w','s','.','S','y','s','t','e','m','.','U','s','e','r','P','r','o','f','i','l','e','.','G','l','o','b','a','l','i','z','a','t','i','o','n','P','r','e','f','e','r','e','n','c','e','s',0};
#elif defined(__GNUC__) && !defined(__cplusplus)
const DECLSPEC_SELECTANY WCHAR RuntimeClass_Windows_System_UserProfile_GlobalizationPreferences[] = L"Windows.System.UserProfile.GlobalizationPreferences";
#else
extern const DECLSPEC_SELECTANY WCHAR RuntimeClass_Windows_System_UserProfile_GlobalizationPreferences[] = {'W','i','n','d','o','w','s','.','S','y','s','t','e','m','.','U','s','e','r','P','r','o','f','i','l','e','.','G','l','o','b','a','l','i','z','a','t','i','o','n','P','r','e','f','e','r','e','n','c','e','s',0};
#endif
#endif /* RUNTIMECLASS_Windows_System_UserProfile_GlobalizationPreferences_DEFINED */
#endif /* WINDOWS_FOUNDATION_UNIVERSALAPICONTRACT_VERSION >= 0x10000 */

/* Begin additional prototypes for all interfaces */

ULONG           __RPC_USER HSTRING_UserSize     (ULONG *, ULONG, HSTRING *);
unsigned char * __RPC_USER HSTRING_UserMarshal  (ULONG *, unsigned char *, HSTRING *);
unsigned char * __RPC_USER HSTRING_UserUnmarshal(ULONG *, unsigned char *, HSTRING *);
void            __RPC_USER HSTRING_UserFree     (ULONG *, HSTRING *);

/* End additional prototypes */

#ifdef __cplusplus
}
#endif

#endif /* __windows_system_userprofile_h__ */
