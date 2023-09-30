__version__=str(1695307673/2**31)
import os,sys
import vapoursynth as vs
from vapoursynth import core
import xvs
import mvsfunc as mvf
import muvsfunc as muf
from functools import partial
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence, Union
import nnedi3_resample as nnrs

'''
functions:
- pqdenoise
- zmdg (zmde)
- xdbcas
- arop
- pfinesharp (rpfilter (rpclip))
- w2xaa
- knl4a (nlm)
- wtfmask
- bordermask
- bm3d (copy-paste!)
- n3pv
- rescale, rescalef, multirescale (copy-paste!)
- quack
- bilateraluv
- dft, idft, dct, idct
- badlyscaledborderdetect
- rescaleandtrytounfuckborders
- isvse, isvspipe
- fmvfps
- hrife
- go444keepuv
- setrange, setmatrix, settransfer, setprimaries, setchromaloc, setparams
'''

try:
    from dotenv import load_dotenv

    if (env := Path(__file__).parent / ".env.zvs.global").exists():
        load_dotenv(env)
    load_dotenv('.env.zvs',override=True)
except ModuleNotFoundError:
    pass

# Override these values if they're found in .env.
nnrs_mode_default = os.environ.get("nnrs_mode_default", "nnedi3")
bm3d_mode_default = os.environ.get("bm3d_mode_default", "cpu")
bm3d_extractor_exp_default = os.environ.get("bm3d_extractor_exp_default", 0)

nnrs.nnedi3_resample=partial(nnrs.nnedi3_resample,mode=nnrs_mode_default,nns=3,nsize=3,qual=2,pscrn=1)
Nnrs=nnrs


#denoise pq hdr content by partially convert it to bt709, do denoise in bt709 then take the difference back to pq, may yield a better result
def pqdenoise(src,sigma=[1,1,1],lumaonly=False,block_step=7,radius=1,finalest=False,bm3dtyp=bm3d_mode_default,vt=0,mdegrain=True,tr=2,pel=1,blksize=16,overlap=None,chromamv=True,thsad=100,thsadc=None,thscd1=400,thscd2=130,truemotion=False,nl=100,contrasharp=1,to709=1,show='output',limit=None,limitc=None,sigma2=None,radius2=None,lf=None,refinemotion=False,rmblksize=None,rmoverlap=None,rmpel=None,rmchromamv=None,rmtruemotion=None,rmthsad=None,pref=None):
    if lumaonly:
        chromamv=False
        chromaclip=src
        src=xvs.getY(src)

    src=src.fmtc.bitdepth(bits=16)
    denoised=sdr=core.resize.Bicubic(src,transfer_in=16,transfer=1,nominal_luminance=nl) if to709 else src
    pref=core.resize.Bicubic(pref,transfer_in=16,transfer=1,nominal_luminance=nl) if pref!=None else pref if to709 else pref
    if mdegrain:
        limitc=limitc if limitc else limit
        denoised=zmde(denoised,tr=tr,thsad=thsad,thsadc=thsadc,blksize=blksize,overlap=overlap,pel=pel,thscd1=thscd1,thscd2=thscd2,truemotion=truemotion,chromamv=chromamv,limit=limit,limitc=limitc,lf=lf,refinemotion=refinemotion,rmblksize=rmblksize,rmoverlap=rmoverlap,rmpel=rmpel,rmchromamv=rmchromamv,rmtruemotion=rmtruemotion,rmthsad=rmthsad,pref=pref)
        if show=='mde':
            return denoised

        if contrasharp>=2 or (contrasharp==1 and bm3dtyp=='no'):
            denoised=ContraSharpening(denoised,sdr)
        if show=='mdecs':
            return denoised

    if not bm3dtyp=='no':
        if vt==0:
            if bm3dtyp=='cpu':
                BM3D=core.bm3dcpu.BM3D
            elif bm3dtyp=='cuda':
                BM3D=core.bm3dcuda.BM3D
            elif bm3dtyp=='cuda_rtc':
                BM3D=core.bm3dcuda_rtc.BM3D
        elif vt==1:
            if bm3dtyp=='cpu':
                BM3D=core.bm3dcpu.BM3Dv2
            elif bm3dtyp=='cuda':
                BM3D=core.bm3dcuda.BM3Dv2
            elif bm3dtyp=='cuda_rtc':
                BM3D=core.bm3dcuda_rtc.BM3Dv2

        bdenoised=BM3D(denoised.fmtc.bitdepth(bits=32),sigma=sigma,radius=radius,block_step=block_step)
        if radius>0 and vt==0:
            bdenoised=core.bm3d.VAggregate(bdenoised,radius,1)

        if finalest:
            sigma2=sigma2 if sigma2 else sigma
            radius2=radius2 if radius2 else radius
            bdenoised=BM3D(denoised.fmtc.bitdepth(bits=32),ref=bdenoised,sigma=sigma2,radius=radius2,block_step=max(block_step-1,1))
            if radius2>0 and vt==0:
                bdenoised=core.bm3d.VAggregate(bdenoised,radius2,1)

        denoised=bdenoised.fmtc.bitdepth(bits=16)
        if show=='bm3d':
            return denoised

    if contrasharp>=1:
        denoised=ContraSharpening(denoised,sdr)
    if show=='bm3dcs':
        return denoised

    if to709:
        denoised,sdr=[core.resize.Bicubic(i,transfer_in=1,transfer=16,nominal_luminance=nl) for i in (denoised,sdr)]

    output=core.std.Expr([src,sdr,denoised],'x y - z +') if to709 else denoised
    if lumaonly:
        output=core.std.ShufflePlanes([output,chromaclip],[0,1,2],vs.YUV)

    return output

#a simple mdegrain wrapper function that's enough for my own use
'''
cs: go
no I'm joking
cs: call ContraSharpening after degrain
mvout: output a dict of mvs ready for "mvin"
mvout_sup: output super clips as well, benefit is small and may cause unintended behavior, not recommended
mvin: take a dict of mvs, use them to degrain, if super clips present, they will be used too
mvinrm: apply recalculate on mvs from "mvin"
mvupd: only with "mvinrm", decide whether to modify the input dict
lf: provide your own func for limit (does not override the "limit" arg of mdegrain) eg: lambda x,y:mvf.LimitFilter(x,y,thr=0.5,elast=20) or a number represents "thr" in mvf.LimitFilter
'''
def zmdg(src,tr=None,thsad=100,thsadc=None,blksize=16,mv_pad=None,resize_pad=True,overlap=None,pel=1,chromamv=True,sharp=2,rfilter=4,dct=0,truemotion=True,thscd1=400,thscd2=130,pref=None,cs=False,csrad=1,csrep=14,cspl=None,refinemotion=False,rmblksize=None,rmoverlap=None,rmpel=None,rmchromamv=None,rmtruemotion=None,rmthsad=None,rmdct=None,mvout=False,mvout_sup=False,mvin=None,mvinrm=False,mvupd=None,lf=None,**args):
    if resize_pad:
        if isinstance(resize_pad,bool):
            src=rpclip(src,blksize)
            if isinstance(pref,vs.VideoNode): pref=rpclip(pref,blksize)
        elif isinstance(resize_pad,int):
            src=rpclip(src,resize_pad)
            if isinstance(pref,vs.VideoNode): pref=rpclip(pref,blksize)
        else:
            raise ValueError
    mvd_in=isinstance(mvin,dict)
    if thsadc==None:
        thsadc=thsad//2
    last=src
    if pref!=None:
        pass
    elif not chromamv:
        pref=core.resize.Bicubic(last,range_in_s='limited',range_s='full')
    else:
        pref=last
    if tr==None:
        if not mvd_in:
            tr=2
        else:
            tr=mvin['tr']
    if overlap==None: overlap=blksize//2
    if rmblksize==None: rmblksize=blksize//2
    if rmoverlap==None: rmoverlap=rmblksize//2
    if rmthsad==None: rmthsad=thsad//2
    if rmpel==None: rmpel=pel
    if rmchromamv==None: rmchromamv=chromamv
    if rmtruemotion==None: rmtruemotion=truemotion
    if rmdct==None: rmdct=dct
    if mvupd==None: mvupd=mvinrm
    if mv_pad==None:
        mv_pad=[blksize]*2+[rmblksize]*2
    elif isinstance(mv_pad,int):
        mv_pad=[mv_pad]*4

    sup,sup2,sup3=0,0,0
    if mvd_in:
        supin=mvin.get('sup')
        if supin!=None:
            sup,sup2=supin[0],supin[1]
            if refinemotion and len(supin)==3:
                sup3=supin[2]

    sup=core.mv.Super(pref,hpad=mv_pad[0],vpad=mv_pad[1],sharp=sharp,rfilter=rfilter,pel=pel) if sup==0 else sup
    sup2=core.mv.Super(last,hpad=mv_pad[0],vpad=mv_pad[1],sharp=sharp,levels=1,pel=pel) if sup2==0 else sup2
    if refinemotion:
        if isinstance(refinemotion,vs.VideoNode): #i can't remember why i did this really, but it's harmless just leave it
            sup3=refinemotion
        else:
            sup3=core.mv.Super(last,hpad=mv_pad[2],vpad=mv_pad[3],sharp=sharp,levels=1,pel=rmpel) if sup3==0 else sup3

    mvfw,mvbw=[],[]
    if mvd_in:
        _mvfw=mvin['mvfw']
        _mvbw=mvin['mvbw']
        if tr>mvin['tr']:raise ValueError
        if mvinrm:
            for i in range(tr):
                _fw=core.mv.Recalculate(sup3,_mvfw[i],rmthsad,blksize=rmblksize,overlap=rmoverlap,truemotion=rmtruemotion,chroma=rmchromamv,dct=rmdct)
                _bw=core.mv.Recalculate(sup3,_mvbw[i],rmthsad,blksize=rmblksize,overlap=rmoverlap,truemotion=rmtruemotion,chroma=rmchromamv,dct=rmdct)
                mvfw.append(_fw)
                mvbw.append(_bw)
                if mvupd:
                    _mvfw[i]=_fw
                    _mvbw[i]=_bw
        else:
            mvfw=_mvfw
            mvbw=_mvbw
    else:
        for i in range(1,tr+1):
            _fw=core.mv.Analyse(sup,isb=False,delta=i,blksize=blksize,overlap=overlap,truemotion=truemotion,chroma=chromamv,dct=dct)
            _bw=core.mv.Analyse(sup,isb=True,delta=i,blksize=blksize,overlap=overlap,truemotion=truemotion,chroma=chromamv,dct=dct)
            if refinemotion:
                _fw=core.mv.Recalculate(sup3,_fw,rmthsad,blksize=rmblksize,overlap=rmoverlap,truemotion=rmtruemotion,chroma=rmchromamv,dct=rmdct)
                _bw=core.mv.Recalculate(sup3,_bw,rmthsad,blksize=rmblksize,overlap=rmoverlap,truemotion=rmtruemotion,chroma=rmchromamv,dct=rmdct)
            mvfw.append(_fw)
            mvbw.append(_bw)

    mv_list_string=','.join([f'mvbw[{j}],mvfw[{j}]' for j in range(tr)])
    if mvout:
        mvd={'mvfw':mvfw,'mvbw':mvbw,'tr':tr,'mvlist':mv_list_string}
        if mvout_sup:
            mvd['sup']=[sup,sup2,sup3] if refinemotion else [sup,sup2]
        return mvd

    last=eval(f'core.mv.Degrain{tr}(last,sup2,{mv_list_string},thsad=thsad,thsadc=thsadc,thscd1=thscd1,thscd2=thscd2,**args)')
    if callable(lf):
        last=lf(last,src)
    elif isinstance(lf,(int,float)):
        last=mvf.LimitFilter(last,src,thr=lf,elast=20)
    if cs:
        last=rpfilter(last,src,filter=lambda x,y: ContraSharpening(x,y,radius=csrad,rep=csrep,planes=cspl),psize=4)
    if resize_pad:
        if isinstance(resize_pad,bool):
            last=last.std.Crop(*[blksize]*4)
        elif isinstance(resize_pad,int):
            last=last.std.Crop(*[resize_pad]*4)
    return last
#for backward compatibility
def zmde(src,thsad=100,thsadc=None,resize_pad=None,truemotion=False,**args):
    if thsadc==None:thsadc=thsad
    return zmdg(src,thsad=thsad,thsadc=thsadc,resize_pad=resize_pad,truemotion=truemotion,**args)

#multi-pass f3kdb with optional contra-sharpening, masking and limit filter
#idea stolen from xyx98
def xdbcas(src,r=[8,15],y=[32,24],cb=[16,10],cr=[16,10],gy=[0,0],gc=[0,0],sm=[2,2],rs=[0,0],bf=[True,True],dg=[False,False],opt=[-1,-1],mt=[True,True],da=[3,3],ktv=[False,False],od=[16,16],rar=[1,1],rag=[1,1],rpr=[1,1],rpg=[1,1],passes=2,neo=True,casstr=0,mask=True,limit=True,s16=False):
    last=db=src.fmtc.bitdepth(bits=16) if s16 else src
    r,y,cb,cr,gy,gc,sm,rs,bf,dg,opt,mt,da,ktv,od,rar,rag,rpr,rpg=[[i]*passes if isinstance(i,int) else i+[i[-1]]*passes for i in (r,y,cb,cr,gy,gc,sm,rs,bf,dg,opt,mt,da,ktv,od,rar,rag,rpr,rpg)]

    # l1,l2,l3,l4,l5,l6=[len(i) for i in (r,y,cb,cr,gy,gc)]
    # if l1==l2==l3==l4==l5==l6:
    #     passes=l6
    # else:
    #     passes=min(l1,l2,l3,l4,l5,l6)

    for i in range(passes):
        if neo:
            db=core.neo_f3kdb.Deband(db,r[i],y[i],cb[i],cr[i],gy[i],gc[i],sm[i],rs[i],bf[i],dg[i],opt[i],mt[i],da[i],ktv[i],od[i],rar[i],rag[i],rpr[i],rpg[i])
        else:
            db=core.f3kdb.Deband(db,r[i],y[i],cb[i],cr[i],gy[i],gc[i],sm[i],rs[i],bf[i],dg[i],opt[i],da[i],ktv[i],od[i],rar[i],rag[i],rpr[i],rpg[i])

    if isinstance(limit,bool) and limit:
        db=mvf.LimitFilter(db,last,thr=0.1,thrc=0.05,elast=20,planes=[0,1,2])
    elif isinstance(limit,(list,tuple)):
        limit=[i for i in limit if isinstance(i,(int,float))]
        if len(limit)==3:
            db=mvf.LimitFilter(db,last,thr=limit[0],thrc=limit[1],elast=limit[2],planes=[0,1,2])
        else:
            raise ValueError
    elif callable(limit):
        db=limit(db,last)
    else:
        pass

    if mask:
        if isinstance(mask,vs.VideoNode):
            dbmask=mask
        else:
            dbmask=xvs.mwdbmask(last)
        db=core.std.MaskedMerge(db,last,dbmask)

    if casstr<=0:
        return db

    cas=core.cas.CAS(last,casstr,planes=[0,1,2])
    cas=mvf.LimitFilter(cas,last,thr=0.3,thrc=0.15,brighten_thr=0.15,elast=4,planes=[0,1,2])
    last=core.std.Expr([cas,last,db],'x y - z +')

    return last

#arbitrary crop, result resolution must be compatible with src clip subsampling tho
def arop(src,left=0,right=0,top=0,bottom=0): #mostly useless experimental function
    subsw,subsh=[2**src.format.subsampling_w,2**src.format.subsampling_h]
    if not (top%subsh or bottom%subsh or left%subsw or right%subsw):
        return core.std.Crop(src,left=left,right=right,top=top,bottom=bottom)
    else:
        l,r=[i if not i%subsw else 0 for i in (left,right)]
        t,b=[i if not i%subsh else 0 for i in (top,bottom)]

        last=core.std.Crop(src,left=l,right=r,top=t,bottom=b) if not(l==r==t==b==0) else src
        y,u,v=xvs.extractPlanes(last)

        l,r=[i if i%subsw else 0 for i in (left,right)]
        t,b=[i if i%subsh else 0 for i in (top,bottom)]

        y=core.std.Crop(y,left=l,right=r,top=t,bottom=b)
        w=(src.width-left-right)//subsw
        h=(src.height-top-bottom)//subsh
        src_left=l/subsw
        src_top=t/subsh
        src_width=(last.width-l-r)/subsw
        src_height=(last.height-t-b)/subsh
        u,v=[core.resize.Bicubic(i,w,h,src_left=src_left,src_top=src_top,src_width=src_width,src_height=src_height) for i in (u,v)]

        last=core.std.ShufflePlanes([y,u,v],[0,0,0],vs.YUV)
        return last

#padded finesharp, because finesharp like to mess up frames' edges
def pfinesharp(src,crop=True,psize=4,**args):
    import finesharp
    sharpen=lambda x: finesharp.sharpen(x,**args)
    return rpfilter(src,filter=sharpen,psize=psize,crop=crop)

#nnrs and ssim is for chroma upscaling and downscaling only
#deprecated, 無駄無駄
def w2xaa(src,model=0,noise=-1,fp32=False,tile_size=0,format=None,full=None,matrix='709',nnrs=False,ssim=False,ssim_smooth=False,ssim_sigmoid=True,nnrs_down=None,ort=False,model_f=None,model_p=None,overlap=None):
    if full==None:
        try:
            full=not src.get_frame(0).props._ColorRange
        except:
            full=False
    src_range_s='full' if full else 'limited'
    src_format=src.format if format==None else format
    width,height=src.width,src.height
    precision=32 if fp32 else 16
    nnrs_down=nnrs if nnrs_down==None else nnrs_down
    if nnrs:
        last=Nnrs.nnedi3_resample(src,csp=vs.RGBS,fulls=full,mats=matrix)
    else:
        last=core.resize.Bicubic(src,format=vs.RGBS,range_in_s=src_range_s,matrix_in_s=matrix)

    if not ort:
        if tile_size==0:
            tile_size=[src.width,src.height]
        elif isinstance(tile_size,int):
            tile_size=[tile_size]*2
        last=core.w2xncnnvk.Waifu2x(last,model=model,scale=2,noise=noise,precision=precision,tile_w=tile_size[0],tile_h=tile_size[1])
    else:
        tile_size=[src.width,src.height] if tile_size==0 else tile_size
        overlap=4 if model==2 else 8 if overlap==None else overlap
        builtin=True if model_f==model_p==None else False
        model_f='waifu2x' if model_f==None else model_f
        model_g=['upconv_7_anime_style_art_rgb','upconv_7_photo','cunet'][model] if isinstance(model,int) else model
        model_n=f'noise{noise}_scale2.0x_model.onnx' if not noise==-1 else 'scale2.0x_model.onnx'
        model_p=os.path.join(model_f,model_g,model_n) if model_p==None else model_p
        last=core.ort.Model(last,model_p,provider='CUDA',builtin=builtin,fp16=not fp32,tilesize=tile_size,overlap=overlap)

    if ssim:
        last=muf.SSIM_downsample(last,width,height,format=src_format,range_s=src_range_s,matrix_s=matrix,smooth=ssim_smooth,sigmoid=ssim_sigmoid)
    elif nnrs_down:
        last=Nnrs.nnedi3_resample(last,width,height,csp=src_format,fulld=full,matd=matrix)
    else:
        last=core.resize.Bicubic(last,width,height,format=src_format,range_s=src_range_s,matrix_s=matrix)
    return last

#a workaround for amd rdna graphic cards to use knlmeanscl
def knl4a(src,*args,**kwargs):
    return nlm(src,*args,**kwargs,amd=True,mode='ocl')
def nlm(src,planes=[1,1,1],rclip=None,h=1.2,amd=False,mode='ocl',**args):
    NLM={'ocl':core.knlm.KNLMeansCL,'cuda':core.nlm_cuda.NLMeans}[mode] if not amd else core.knlm.KNLMeansCL
    if isinstance(h,list):
        if len(h)>=3:
            pass
        elif len(h)==2:
            h=[h[0],h[1],h[1]]
        else:
            h=[h[0],h[0],h[0]]
    else:
        h=[h,h,h]

    if not amd and planes==[1,1,1] and h[1]==h[2]:
        return NLM(NLM(src,rclip=rclip,h=h[0],channels='Y',**args),rclip=rclip,h=h[1],channels='UV',**args)

    y,u,v=xvs.extractPlanes(src)
    if amd:
        y,u,v=[core.std.ShufflePlanes([i,i,i],[0,0,0],vs.RGB) for i in (y,u,v)]
    if isinstance(rclip,vs.VideoNode):
        ry,ru,rv=[i(rclip) for i in (xvs.getY,xvs.getU,xvs.getV)]
        if amd:
            ry,ru,rv=[core.std.ShufflePlanes([i,i,i],[0,0,0],vs.RGB) for i in (ry,ru,rv)]
    else:
        ry=ru=rv=None
    if planes[0]:
        y=NLM(y,rclip=ry,h=h[0],**args)
    if planes[1] and planes[2] and h[1]==h[2]:
        uv=NLM(src,rclip=rclip,h=h[1],channels='UV',**args)
        u=xvs.getU(uv)
        v=xvs.getV(uv)
    else:
        if planes[1]:
            u=NLM(u,rclip=ru,h=h[1],**args)
        if planes[2]:
            v=NLM(v,rclip=rv,h=h[2],**args)
    y,u,v=[core.std.ShufflePlanes(i,[0],vs.GRAY) for i in (y,u,v)]
    return core.std.ShufflePlanes([y,u,v],[0,0,0],vs.YUV)

#line mask?
def wtfmask(src,nnrs=True,t_l=16,t_h=26,range='full',op=[1],optc=1,bin=True,bthr=1,**args):
    if nnrs:
        last=Nnrs.nnedi3_resample(src,csp=vs.RGBS)
    else:
        last=core.resize.Bicubic(src,format=vs.RGBS,matrix_in=1)
    last=core.tcanny.TCanny(last,t_l=t_l,t_h=t_h,op=optc,**args)
    if range in ['full','pc']:
        last=last.resize.Bicubic(format=vs.GRAY16,matrix=1,range_s='full')
        if bin: last=last.std.Binarize(256*bthr,0,65535)
    else:
        last=last.resize.Bicubic(format=vs.GRAY16,matrix=1,range_s='limited')
        if bin: last=last.std.Binarize(256*(16+(235-16)/256*bthr),16*256,235*256)
    f=[core.std.Minimum,core.std.Maximum,core.std.Deflate,core.std.Inflate]
    for i in op:
        last=f[i](last)
    return last


def bordermask(src,l=0,r=0,t=0,b=0,d=16):
    return core.std.BlankClip(src,format=core.std.BlankClip(format=vs.GRAY16).fmtc.bitdepth(bits=d).format,color=0).std.Crop(left=l,right=r,top=t,bottom=b).std.AddBorders(left=l,right=r,top=t,bottom=b,color=2**d-1 if d<32 else 1)

#resize padded filter
def rpfilter(input,ref=None,other=None,filter=lambda x: x,psize=2,crop=True):
    input=rpclip(input,psize)
    if isinstance(other,dict):
        for key in other:
            if isinstance(other[key],vs.VideoNode):
                other[key]=rpclip(other[key],psize)
    if isinstance(ref,vs.VideoNode):
        ref=rpclip(ref,psize)
        if isinstance(other,dict):
            last=filter(input,ref,**other)
        else:
            last=filter(input,ref)
    else:
        if isinstance(other,dict):
            last=filter(input,**other)
        else:
            last=filter(input)
    if crop:
        last=core.std.Crop(last,*[psize]*4)
    return last

#resize pad clip
def rpclip(input,psize=2,left=None,right=None,top=None,bottom=None):
    if left==None:left=psize
    if right==None:right=psize
    if top==None:top=psize
    if bottom==None:bottom=psize
    w=input.width+left+right
    h=input.height+top+bottom
    return core.resize.Bicubic(input,w,h,src_top=-top,src_left=-left,src_width=w,src_height=h)

#nnedi3 preview
def n3pv(*args,**kwargs):
    scale=kwargs.get('scale') if kwargs.get('scale')!=None else 2
    nns=kwargs.get('nns') if kwargs.get('nns')!=None else 1
    nsize=kwargs.get('nsize') if kwargs.get('nsize')!=None else 0
    qual=kwargs.get('qual') if kwargs.get('qual')!=None else 1
    mode=kwargs.get('mode') if kwargs.get('mode')!=None else nnrs_mode_default
    int_=kwargs.get('int') if kwargs.get('int')!=None else True
    depth=kwargs.get('depth') if kwargs.get('depth')!=None else 8
    mats=kwargs.get('mats')
    fulls=kwargs.get('fulls')
    bypass=mode=='bypass'
    csp=eval(f'vs.RGB{depth*3}')
    last=list()
    if len(args)==1:
        if isinstance(args[0],list):
            mats=[mats]*len(args[0]) if not isinstance(mats,(list,tuple)) else mats
            fulls=[fulls]*len(args[0]) if not isinstance(fulls,(list,tuple)) else fulls
            for i,clip in enumerate(args[0]):
                if bypass:
                    _tmpclp=core.resize.Bicubic(clip,clip.width*scale,clip.height*scale)
                    _tmpclp=mvf.ToRGB(_tmpclp,depth=depth,matrix=mats[i],full=fulls[i])
                else:
                    _tmpclp=Nnrs.nnedi3_resample(clip,clip.width*scale,clip.height*scale,csp=csp,nns=nns,nsize=nsize,qual=qual,mode=mode,mats=mats[i],fulls=fulls[i])
                last.append(_tmpclp)
        elif isinstance(args[0],vs.VideoNode):
            mats=mats[0] if isinstance(mats,(list,tuple)) else mats
            fulls=fulls[0] if isinstance(fulls,(list,tuple)) else fulls
            if bypass:
                _tmpclp=core.resize.Bicubic(args[0],args[0].width*scale,args[0].height*scale)
                _tmpclp=mvf.ToRGB(_tmpclp,depth=depth,matrix=mats,full=fulls)
            else:
                _tmpclp=Nnrs.nnedi3_resample(args[0],args[0].width*scale,args[0].height*scale,csp=csp,nns=nns,nsize=nsize,qual=qual,mode=mode,mats=mats,fulls=fulls)
            last.append(_tmpclp)
        else:
            raise TypeError('input for preview should be list or clip')
    else:
        mats=[mats]*len(args) if not isinstance(mats,(list,tuple)) else mats
        fulls=[fulls]*len(args) if not isinstance(fulls,(list,tuple)) else fulls
        for i,clip in enumerate(args):
            if bypass:
                _tmpclp=core.resize.Bicubic(clip,clip.width*scale,clip.height*scale)
                _tmpclp=mvf.ToRGB(_tmpclp,depth=depth,matrix=mats[i],full=fulls[i]).sub.Subtitle('clip%d'%i)
            else:
                _tmpclp=Nnrs.nnedi3_resample(clip,clip.width*scale,clip.height*scale,csp=csp,nns=nns,nsize=nsize,qual=qual,mode=mode,mats=mats[i],fulls=fulls[i]).sub.Subtitle('clip%d'%i)
            last.append(_tmpclp)
    return last[0] if len(last)==1 else core.std.Interleave(last) if int_ else last

#quack quack, I'll take your grains
#a dumb-ass func may be suitable for old movies with heavy dynamic grains
def quack(src,bilateral=False,median=None,knl={},md1={},bm1={},md2={},bm2={}):
    _knl={'amd':False,'a':1,'s':2,'d':3,'h':2}
    _md1={'thsad':250,'thscd1':250,'limit':768,'tr':3}
    _bm1={'sigma':[2,2,2],'radius':1}
    _md2={'thsad':250,'thscd1':250,'limit':768,'tr':3}
    _bm2={'sigma':[1,1,1],'radius':1}
    if median is None:
        median=knl!=False
    if bilateral is None:
        if knl!=False:bilateral='cpu'
    if not knl==False:
        _knl.update(knl)
    _md1.update(md1)
    _bm1.update(bm1)
    _md2.update(md2)
    _bm2.update(bm2)
    if src.format.bits_per_sample!=16:
        src=src.fmtc.bitdepth(bits=16)
    last=src
    if bilateral:
        m=last.bilateralgpu.Bilateral() if bilateral=='gpu' else last.bilateral.Bilateral(planes=[0,1,2])
    elif median:
        m=last.std.Median()
    else:
        m=src
    if knl==False:
        n=m
    else:
        n=nlm(last,rclip=m,**_knl)
    last=zmdg(last,pref=n,**_md1)
    last=bm3d(last,iref=src,**_bm1)
    last=zmdg(src,pref=last,**_md2)
    last=bm3d(last,iref=src,**_bm2)
    return last

#use y channel or opponent chroma channel as reference to repair uv channels with bilateral
#tbilateral is much trickier to use, the "ref" doesn't even mean the same thing, just add it for testing
#parameters you should really care about are ones in first line
#using KrigBilateral with vs-placebo may be better?
def bilateraluv(src,ch='uv',mode='down',method='spline36',oldbehavior=False,clc=True,left=True,top=False,S=1,R=0.02,lumaref=True,crossref=False,\
    algo=0,P=None,T=False,diameter=3,sdev=0.5,idev=0.01,cs=1,d2=True,kerns=1,kerni=1,restype=0,**kwargs):
    if mode.lower()=='up':
        targetw=src.width
        targeth=src.height
    elif mode.lower()=='down':
        targetw=src.width>>src.format.subsampling_w
        targeth=src.height>>src.format.subsampling_h
    else:
        raise ValueError('mode not supported')

    if lumaref:
        if method.lower()=='nnrs':
            resizer=lambda x,w,h,l,t,**args:Nnrs.nnedi3_resample(x,w,h,src_left=l,src_top=t,csp=args.pop('format') if oldbehavior else None,**args)
        elif method.lower() in ['point','bilinear','bicubic','lanczos','spline16','spline36','spline64']:
            resizer=eval(f'lambda x,w,h,l,t,**args:core.resize.{method.capitalize()}(x,w,h,src_left=l,src_top=t,**args)')
        else:
            raise ValueError('resize method not supported')
        last=go444keepuv(src,dir=mode,clc=clc,left=left,top=top,resampler=partial(resizer,**kwargs)) if not oldbehavior else resizer(src,targetw,targeth,0,0,format=vs.YUV444P16,**kwargs)
    else:
        last=src

    y,u,v=xvs.extractPlanes(last)
    if 'u' in ch.lower():
        if lumaref:
            ub=core.bilateral.Bilateral(u,y,sigmaS=S,sigmaR=R,algorithm=algo,PBFICnum=P) if not T else\
                core.tbilateral.TBilateral(u,y,diameter,sdev,idev,cs,d2,kerns,kerni,restype)
        else:
            ub=u
        if crossref:
            ub=core.bilateral.Bilateral(ub,v,sigmaS=S,sigmaR=R,algorithm=algo,PBFICnum=P) if not T else\
                core.tbilateral.TBilateral(ub,v,diameter,sdev,idev,cs,d2,kerns,kerni,restype)
    if 'v' in ch.lower():
        if lumaref:
            vb=core.bilateral.Bilateral(v,y,sigmaS=S,sigmaR=R,algorithm=algo,PBFICnum=P) if not T else\
                core.tbilateral.TBilateral(v,y,diameter,sdev,idev,cs,d2,kerns,kerni,restype)
        else:
            vb=v
        if crossref:
            vb=core.bilateral.Bilateral(vb,u,sigmaS=S,sigmaR=R,algorithm=algo,PBFICnum=P) if not T else\
                core.tbilateral.TBilateral(vb,u,diameter,sdev,idev,cs,d2,kerns,kerni,restype)
    return core.std.ShufflePlanes([src,ub,vb],[0,0,0],vs.YUV)

#this is cool but I don't really know why I wrote this
#some reference:
#https://stackoverflow.com/questions/59975604/how-to-inverse-a-dft-with-magnitude-with-opencv-python
def dft(src,d=10,spectrum=False,split=True,linear='1886',shift=True):
    if src.format.id != vs.GRAYS:
        raise ValueError('I thought only GRAYS input was supported.')
    import numpy as np
    import cv2
    if linear is not False:
        src=src.fmtc.transfer(transs=linear,transd='linear')
    src=src.std.PlaneStats()
    def dft(n,f,h):
        fout=f.copy()
        Dft=np.asarray(fout[0])[:h,:]
        Dft=cv2.dft(Dft,flags=cv2.DFT_COMPLEX_OUTPUT)
        if shift: Dft=np.fft.fftshift(Dft)
        mag,phase=cv2.cartToPolar(Dft[:,:,0],Dft[:,:,1])
        if spectrum:
            spec=np.log(mag)/d
            Stack=np.concatenate([mag,phase,spec])
        else:
            Stack=np.concatenate([mag,phase])
        np.copyto(np.asarray(fout[0]),Stack)
        return fout
    stack=core.std.StackVertical([src]*(2+spectrum))
    stack=core.std.ModifyFrame(stack,stack,partial(dft,h=src.height))
    if not split:
        return stack
    mag=stack.std.Crop(bottom=src.height*(1+spectrum))
    phase=stack.std.Crop(top=src.height,bottom=src.height*(0+spectrum))
    if spectrum:
        spec=stack.std.Crop(top=src.height*2)
        return mag,phase,spec
    return mag,phase


def idft(mag,phase,linear='1886',ishift=True):
    if not (mag.format.id==phase.format.id==vs.GRAYS):
        raise ValueError('I thought only GRAYS inputs were supported.')
    import numpy as np
    import cv2
    def idft(n,f):
        fout=f[0].copy()
        mag=np.asarray(f[0].copy()[0])
        phase=np.asarray(f[1].copy()[0])
        real,imag=cv2.polarToCart(mag,phase)
        fr=cv2.merge([real,imag])
        if ishift: fr=np.fft.ifftshift(fr)
        fr=cv2.idft(fr)
        fr=cv2.magnitude(fr[:,:,0],fr[:,:,1])
        fr=cv2.normalize(fr,None,f[0].props.PlaneStatsMin,f[0].props.PlaneStatsMax,cv2.NORM_MINMAX,dtype=cv2.CV_32F)
        np.copyto(np.asarray(fout[0]),fr)
        return fout
    last=core.std.ModifyFrame(mag,[mag,phase],idft)
    if linear is not False:
        last=last.fmtc.transfer(transs='linear',transd=linear)
    return last


def dct(src):
    if not src.format.id==vs.GRAYS:
        raise ValueError('I thought only GRAYS input was supported.')
    import numpy as np
    import cv2
    def dct(n,f):
        fout=f.copy()
        tr=np.asarray(fout[0])
        tr=cv2.dct(tr)
        np.copyto(np.asarray(fout[0]),tr)
        return fout
    return core.std.ModifyFrame(src,src,dct)


def idct(src):
    if not src.format.id==vs.GRAYS:
        raise ValueError('I thought only GRAYS input was supported.')
    import numpy as np
    import cv2
    def idct(n,f):
        fout=f.copy()
        tr=np.asarray(fout[0])
        tr=cv2.idct(tr)
        np.copyto(np.asarray(fout[0]),tr)
        return fout
    return core.std.ModifyFrame(src,src,idct)

#I don't even know what this do, will likely be abandoned
def dwt(src,w='Haar'):
    if not src.format.id==vs.GRAYS:
        raise ValueError('I thought only GRAYS input was supported.')
    import numpy as np
    import pywt
    def dwt(n,f,w):
        fout=f.copy()
        tr=np.asarray(fout[0])
        a,(h,v,d)=pywt.dwt2(tr,w)
        tr=np.concatenate([np.concatenate([a,v]),np.concatenate([h,d])],axis=1)
        np.copyto(np.asarray(fout[0]),np.float32(tr))
        return fout
    return core.std.ModifyFrame(src,src,partial(dwt,w=w))


def idwt(src,w='Haar'):
    if not src.format.id==vs.GRAYS:
        raise ValueError('I thought only GRAYS input was supported.')
    import numpy as np
    import pywt
    def dwt(n,f,w):
        fout=f.copy()
        tr=np.asarray(fout[0])
        a1,a2=np.hsplit(tr,2)
        (a,v),(h,d)=np.vsplit(a1,2),np.vsplit(a2,2)
        tr=pywt.idwt2([a,(h,v,d)],w)
        np.copyto(np.asarray(fout[0]),np.float32(tr))
        return fout
    return core.std.ModifyFrame(src,src,partial(dwt,w=w))

#badly scaled border detect
def badlyscaledborderdetect(src,left=True,right=True,top=True,bottom=True,conditionmode='or',valuemode='avg',showfrac=False,thr=0.9,thrmode='<',leftline1pos=0,leftline2pos=1,rightline1pos=0,rightline2pos=1,topline1pos=0,topline2pos=1,bottomline1pos=0,bottomline2pos=1):
    cliplist=[src]
    luma=xvs.getY(src)
    def sel1(n,f):
        fout=f[0].copy()
        try:
            if valuemode=='avg':
                fout.props.frac=f[0].props.PlaneStatsAverage/f[1].props.PlaneStatsAverage
            elif valuemode=='max':
                fout.props.frac=f[0].props.PlaneStatsMax/f[1].props.PlaneStatsMax
            elif valuemode=='min':
                fout.props.frac=f[0].props.PlaneStatsMin/f[1].props.PlaneStatsMin
        except ZeroDivisionError:
            fout.props.frac=1
        return fout
    letters=' '
    if left:
        ll1=core.std.Crop(luma,left=leftline1pos,right=luma.width-1-leftline1pos).std.PlaneStats()
        ll2=core.std.Crop(luma,left=leftline2pos,right=luma.width-1-leftline2pos).std.PlaneStats()
        lf=core.std.ModifyFrame(ll1,[ll1,ll2],sel1)
        cliplist.append(lf)
        letters+='L'
    if right:
        rl1=core.std.Crop(luma,right=rightline1pos,left=luma.width-1-rightline1pos).std.PlaneStats()
        rl2=core.std.Crop(luma,right=rightline2pos,left=luma.width-1-rightline2pos).std.PlaneStats()
        rf=core.std.ModifyFrame(rl1,[rl1,rl2],sel1)
        cliplist.append(rf)
        letters+='R'
    if top:
        tl1=core.std.Crop(luma,top=topline1pos,bottom=luma.height-1-topline1pos).std.PlaneStats()
        tl2=core.std.Crop(luma,top=topline2pos,bottom=luma.height-1-topline2pos).std.PlaneStats()
        tf=core.std.ModifyFrame(tl1,[tl1,tl2],sel1)
        cliplist.append(tf)
        letters+='T'
    if bottom:
        bl1=core.std.Crop(luma,bottom=bottomline1pos,top=luma.height-1-bottomline1pos).std.PlaneStats()
        bl2=core.std.Crop(luma,bottom=bottomline2pos,top=luma.height-1-bottomline2pos).std.PlaneStats()
        bf=core.std.ModifyFrame(bl1,[bl1,bl2],sel1)
        cliplist.append(bf)
        letters+='B'
    def sel2(n,f,thr=thr):
        fout=f[0].copy()
        ftot=len(f)
        if showfrac: fracs=[];fracs_in_thr=[]
        if conditionmode=='or':
            isbad=False
            for i in range(1,ftot):
                isbad=isbad or eval(f'f[i].props.frac{thrmode}thr')
                if showfrac:
                    fracs.append(f'{letters[i]}?:{f[i].props.frac}')
                    if eval(f'f[i].props.frac{thrmode}thr'):
                        fracs_in_thr.append(f'{letters[i]}?:{f[i].props.frac}')
        elif conditionmode=='and':
            isbad=True
            for i in range(1,ftot):
                isbad=isbad and eval(f'f[i].props.frac{thrmode}thr')
                if showfrac:
                    fracs.append(f'{letters[i]}?:{f[i].props.frac}')
                    if eval(f'f[i].props.frac{thrmode}thr'):
                        fracs_in_thr.append(f'{letters[i]}?:{f[i].props.frac}')
        elif isinstance(conditionmode,int):
            countbad=0
            for i in range(1,ftot):
                countbad+=eval(f'f[i].props.frac{thrmode}thr')
                if showfrac:
                    fracs.append(f'{letters[i]}?:{f[i].props.frac}')
                    if eval(f'f[i].props.frac{thrmode}thr'):
                        fracs_in_thr.append(f'{letters[i]}?:{f[i].props.frac}')
            isbad=countbad>=conditionmode
        fout.props.badborder=isbad
        if showfrac:
            fout.props.borderfracs=', '.join(fracs)
            fout.props.borderfracs_in_thr=', '.join(fracs_in_thr)
        return fout
    return core.std.ModifyFrame(src,cliplist,sel2)
#why, past me, why?
bsbd=badlyscaledborderdetect

#rescale and try to unfuck border, target on highly specific situation
#ALWAYS DO TESTS BEFORE USE!
def rescaleandtrytounfuckborders(src,w=None,h=None,mask=True,mopf=None,mask_gen_clip=None,mask_dif_pix=2.5,kernel='bilinear',b=0,c=0.5,taps=3,nns=3,nsize=3,qual=2,pscrn=1,show='result',offst1=1,offsl1=1,offst2=1/3,offsl2=1/3,cuth=1,cutv=1,down_kernel=None,post_kernel='bicubic',nns2=None,nsize2=None,qual2=None,pscrn2=None,rim=64,border=4,bc=1,rc=3,linear=False,sigmoid=False,custom_nnedi3down=False,**args):
    if src.format.bits_per_sample!=16:src=src.fmtc.bitdepth(bits=16)
    last=src
    srcw,srch=src.width,src.height
    isgray=last.format.color_family==vs.GRAY
    if w==None and h==None:raise ValueError
    if w==None and isinstance(h,int):w=int(h*(srcw/srch))
    if isinstance(w,int) and h==None:h=int(w*(srch/srcw))
    if nns2==None:nns2=nns
    if nsize2==None:nsize2=nsize
    if qual2==None:qual2=qual
    if pscrn2==None:pscrn2=pscrn
    if mopf==None:mopf=lambda x:xvs.inpand(xvs.expand(x,cycle=2),cycle=2)
    post_kernel=eval(f'core.resize.{post_kernel.capitalize()}') if not callable(post_kernel) else post_kernel
    down_kernel=(post_kernel if not callable(custom_nnedi3down) else custom_nnedi3down) if down_kernel==None else down_kernel
    down_kernel=eval(f'core.resize.{down_kernel.capitalize()}') if not callable(down_kernel) else down_kernel
    tin='1886' if args.get("tin") is None else args.get("tin")
    fulls=False if args.get("fulls") is None else args.get("fulls")
    fulld=True if args.get("fulld") is None else args.get("fulld")

    luma=xvs.getY(last) if not isgray else last
    if isinstance(mask_gen_clip,vs.VideoNode):
        luma=core.std.Interleave([luma,xvs.getY(mask_gen_clip)])
    if sigmoid:
        luma=core.fmtc.transfer(luma,transs=tin,transd='sigmoid',fulls=fulls,fulld=fulld)
    elif linear:
        luma=core.fmtc.transfer(luma,transs=tin,transd='linear',fulls=fulls,fulld=fulld)
    luma32=luma.fmtc.bitdepth(bits=32)

    # luma_de=eval(f'core.descale.De{kernel.lower()}(luma.fmtc.bitdepth(bits=32),{w},{h},src_top=-offst1,src_left=-offsl1)')
    # luma_de2=eval(f'core.descale.De{kernel.lower()}(luma.fmtc.bitdepth(bits=32),{w},{h},src_top=offst1,src_left=offsl1)')
    luma_de=core.descale.Descale(luma32,w,h,kernel=kernel,b=b,c=c,taps=taps,src_top=-offst1,src_left=-offsl1)
    luma_de2=core.descale.Descale(luma32,w,h,kernel=kernel,b=b,c=c,taps=taps,src_top=offst1,src_left=offsl1)
    resize_params=f'filter_param_a={b},filter_param_b={c},'if kernel=='bicubic'else f'filter_param_a={taps},'if kernel=='lanczos'else''
    luma_up=eval(f'core.resize.{kernel.capitalize()}(luma_de,{srcw},{srch},{resize_params}src_top=-offst1,src_left=-offsl1).fmtc.bitdepth(bits=16,dmode=1)')

    if isinstance(mask_gen_clip,vs.VideoNode):
        mclip=xvs.getY(mask_gen_clip)
        mclip_up=luma_up[1::2]
        luma=luma[::2]
        luma_de=luma_de[::2]
        luma_de2=luma_de2[::2]
        luma_up=luma_up[::2]
    ###
    blk=core.std.BlankClip(luma_de,color=0)
    bmask=bordermask(blk,*[rim]*4,32)
    luma_de1=luma_de.std.Crop(right=cuth,bottom=cutv).std.AddBorders(left=cuth,top=cutv,color=0)
    luma_de2=luma_de2.std.Crop(left=cuth,top=cutv).std.AddBorders(right=cuth,bottom=cutv,color=0)
    luma_de1=core.std.MaskedMerge(blk,luma_de1,bmask)
    luma_de2=core.std.MaskedMerge(blk,luma_de2,bmask)

    if sigmoid:
        luma_de=core.fmtc.transfer(luma_de.fmtc.bitdepth(bits=16),transs='sigmoid',transd=tin,fulls=fulld,fulld=fulls)
        luma_de1=core.fmtc.transfer(luma_de1.fmtc.bitdepth(bits=16),transs='sigmoid',transd=tin,fulls=fulld,fulld=fulls)
        luma_de2=core.fmtc.transfer(luma_de2.fmtc.bitdepth(bits=16),transs='sigmoid',transd=tin,fulls=fulld,fulld=fulls)
    elif linear:
        luma_de=core.fmtc.transfer(luma_de.fmtc.bitdepth(bits=16),transs='linear',transd=tin,fulls=fulld,fulld=fulls)
        luma_de1=core.fmtc.transfer(luma_de1.fmtc.bitdepth(bits=16),transs='linear',transd=tin,fulls=fulld,fulld=fulls)
        luma_de2=core.fmtc.transfer(luma_de2.fmtc.bitdepth(bits=16),transs='linear',transd=tin,fulls=fulld,fulld=fulls)

    luma_rescale1=Nnrs.nnedi3_dh(luma_de1,mode=nnrs_mode_default,nns=nns2,nsize=nsize2,qual=qual2,pscrn=pscrn2,field=0).std.Transpose()
    luma_rescale1=Nnrs.nnedi3_dh(luma_rescale1,mode=nnrs_mode_default,nns=nns2,nsize=nsize2,qual=qual2,pscrn=pscrn2,field=0).std.Transpose()
    luma_rescale1=down_kernel(luma_rescale1,srcw,srch)
    luma_rescale2=Nnrs.nnedi3_dh(luma_de2,mode=nnrs_mode_default,nns=nns2,nsize=nsize2,qual=qual2,pscrn=pscrn2,field=1).std.Transpose()
    luma_rescale2=Nnrs.nnedi3_dh(luma_rescale2,mode=nnrs_mode_default,nns=nns2,nsize=nsize2,qual=qual2,pscrn=pscrn2,field=1).std.Transpose()
    luma_rescale2=down_kernel(luma_rescale2,srcw,srch)
    luma_rescale1=post_kernel(luma_rescale1,src_left=offsl2,src_top=offst2,src_width=srcw,src_height=srch)
    luma_rescale2=post_kernel(luma_rescale2,src_left=-offsl2,src_top=-offst2,src_width=srcw,src_height=srch)
    luma_edge=core.std.MaskedMerge(luma_rescale1,luma_rescale2,bordermask(luma,t=border,l=border,d=32)).fmtc.bitdepth(bits=16)

    miss_mask=core.akarin.Expr(luma,f"X {srcw-border-1} > Y {border} < and X {border} < Y {srch-border-1} > and or  65535 0 ?")
    ###
    if callable(custom_nnedi3down):
        luma_rescale=Nnrs.nnedi3_resample(luma_de,luma_de.width*2,luma_de.height*2,qual=qual,nsize=nsize,nns=nns,pscrn=pscrn,src_top=-offst1,src_left=-offsl1)
        luma_rescale=custom_nnedi3down(luma_rescale,srcw,srch).fmtc.bitdepth(bits=16)
    else:
        luma_rescale=Nnrs.nnedi3_resample(luma_de,srcw,srch,qual=qual,nsize=nsize,nns=nns,pscrn=pscrn,src_top=-offst1,src_left=-offsl1).fmtc.bitdepth(bits=16)
    luma_rescale=core.std.MaskedMerge(luma_rescale,luma_edge,bordermask(luma,*[border]*4))

    luma_fixedge=core.edgefixer.Continuity(luma,left=bc,right=bc,top=bc,bottom=bc,radius=rc)
    luma_rescale=core.std.MaskedMerge(luma_rescale,luma_fixedge,miss_mask)

    if mask:
        if not isinstance(mask_gen_clip,vs.VideoNode):
            mclip,mclip_up=luma,luma_up
        mask=core.std.Expr([mclip,mclip_up],"x y - abs").std.Binarize(mask_dif_pix*256)
        mask=core.std.Crop(mask,*[border]*4).std.AddBorders(*[border]*4,color=0)
        mask=mopf(mask)
    if show=='mask': return mask
    if show=='miss_mask': return miss_mask
    if show=='descale': return luma_de
    if show=='descale1': return luma_de1
    if show=='descale2': return luma_de2

    luma_rescale=core.std.MaskedMerge(luma_rescale,luma,mask)
    last=core.std.ShufflePlanes([luma_rescale,last],[0,1,2],vs.YUV) if not isgray else luma_rescale
    return last
#sorry
rattub=rescaleandtrytounfuckborders

#for no reason
def isvse():
    return sys.executable.find('vsedit')!=-1
def isvspipe():
    return sys.executable.find('vspipe')!=-1

#fake mvtools fps
def fmvfps(src,num=60,den=1,blend=True):
    _fn=src.num_frames
    _d=core.std.BlankClip(src,length=1)
    sup=core.mv.Super(_d,pel=1,levels=1)*_fn
    mv=core.mv.Analyse(sup,isb=True,truemotion=False,levels=1)[0]*_fn
    mv2=core.mv.Analyse(sup,truemotion=False,levels=1)[0]*_fn
    return core.mv.BlockFPS(src,sup,mv,mv2,num,den,mode=0,blend=blend)

#it's a helper
def hrife(src,ref=None,mode=None,m='709',format=None,rgbh=True):
    from math import ceil
    if ref is None:ref=src
    if format is None:format=ref.format
    s_w,s_h=ref.width,ref.height
    p_w,p_h=ceil(s_w/32)*32,ceil(s_h/32)*32
    if mode=='i' or src.format.color_family==vs.YUV:
        return src.resize.Bicubic(p_w,p_h,src_width=p_w,src_height=p_h,format=[vs.RGBS,vs.RGBH][rgbh],matrix_in_s=m)
    elif mode=='o' or src.format.color_family==vs.RGB:
        return src.resize.Bicubic(s_w,s_h,src_width=s_w,src_height=s_h,format=format,matrix_s=m)

#can be used in fmtc
#pointless as we already have yuv (typically), I thought this is only meaningful when dealing with RGB like in the literature (for somewhat faster conversion than traditional matrix maybe) so the situation is like yuv (including opp) is better than RGB but which *reasonable* matrix you use is much less relevant, based on my solely observations though
crgb2opp=[1/3,1/3,1/3,0,
1/2,-1/2,0,0,
1/4,1/4,-1/2,0]
copp2rgb=[1,1,2/3,0,
1,-1,2/3,0,
1,0,-4/3,0]


def go444keepuv(src,dir='down',clc=True,left=True,top=False,resampler=None):
    sw,sh=src.width,src.height
    ssw,ssh=src.format.subsampling_w,src.format.subsampling_h
    tw,th=sw>>ssw,sh>>ssh
    if resampler is None:resampler=lambda x,w,h,l,t:core.resize.Spline36(x,w,h,src_left=l,src_top=t)
    if dir.lower()=='down':
        luma=xvs.getY(src)
        luma=resampler(luma,tw,th,0,0) if not clc else resampler(luma,tw,th,-ssw/2 if left else 0,-ssh/2 if top else 0)
        return core.std.ShufflePlanes([luma,src],[0,1,2],vs.YUV)
    elif dir.lower()=='up':
        luma,u,v=xvs.extractPlanes(src)
        u=resampler(u,sw,sh,0,0)
        v=resampler(v,sw,sh,0,0)
        if clc:luma=resampler(luma,sw,sh,-ssw/2 if left else 0,-ssh/2 if top else 0)
        return core.std.ShufflePlanes([luma,u,v],[0,0,0],vs.YUV)


def setrange(src,range):
    if range in ('remove','rm','del'):
        return src.std.RemoveFrameProps('_ColorRange')
    if range in ('full','pc','jpeg'):
        rangeval=0
    elif range in ('limited','tv','mpeg'):
        rangeval=1
    else:
        raise ValueError
    return src.std.SetFrameProps(_ColorRange=rangeval)


def setmatrix(src,matrix):
    if matrix in ('remove','rm','del'):
        return src.std.RemoveFrameProps('_Matrix')
    if isinstance(matrix,int) and matrix>=0 and matrix<256:
        return src.std.SetFrameProps(_Matrix=matrix)
    matrixvaldict={'rgb':0,'709':1,'unspec':2,'fcc':4,'470bg':5,'170m':6,'240m':7,'ycgco':8,'2020ncl':9,'2020cl':10,'ydzdx':11,'chromancl':12,'chromacl':13,'ictcp':14,'601':5,709:1,601:5}
    matrixval=matrixvaldict.get(matrix)
    if matrixval==None:
        raise ValueError
    return src.std.SetFrameProps(_Matrix=matrixval)


def settransfer(src,transfer):
    if transfer in ('remove','rm','del'):
        return src.std.RemoveFrameProps('_Transfer')
    if isinstance(transfer,int) and transfer>=0 and transfer<256:
        return src.std.SetFrameProps(_Transfer=transfer)
    transfervaldict={'709':1,'unspec':2,'470m':4,'470bg':5,'601':6,'240m':7,'linear':8,'log100':9,'log316':10,'xvycc':11,'srgb':13,'2020_10':14,'2020_12':15,'st2084':16,'st428':17,'std-b67':18,'2084':16,'pq':16,'hlg':18,709:1,601:6,2084:16}
    transferval=transfervaldict.get(transfer)
    if transferval==None:
        raise ValueError
    return src.std.SetFrameProps(_Transfer=transferval)


def setprimaries(src,primaries):
    if primaries in ('remove','rm','del'):
        return src.std.RemoveFrameProps('_Primaries')
    if isinstance(primaries,int) and primaries>=0 and primaries<256:
        return src.std.SetFrameProps(_Primaries=primaries)
    primariesvaldict={'709':1,'unspec':2,'470m':4,'470bg':5,'170m':6,'240m':7,'film':8,'2020':9,'st428':10,'xyz':10,'st431-2':11,'st432-1':12,'jedec-p22':22,'p3dci':11,'p3d65':12,709:1,2020:9}
    primariesval=primariesvaldict.get(primaries)
    if primariesval==None:
        raise ValueError
    return src.std.SetFrameProps(_Primaries=primariesval)


def setchromaloc(src,chromaloc):
    if chromaloc in ('remove','rm','del'):
        return src.std.RemoveFrameProps('_ChromaLocation')
    if isinstance(chromaloc,int) and chromaloc>=0 and chromaloc<6:
        return src.std.SetFrameProps(_ChromaLocation=chromaloc)
    chromalocvaldict={'left':0,'center':1,'centre':1,'topleft':2,'top left':2,'top':3,'bottomleft':4,'bottom left':4,'bottom':5}
    chromalocval=chromalocvaldict.get(chromaloc)
    if chromalocval==None:
        return ValueError
    return src.std.SetFrameProps(_ChromaLocation=chromalocval)


def setparams(src,range=None,matrix=None,transfer=None,primaries=None,chromaloc=None):
    if range is not None:
        src=setrange(src,range)
    if matrix is not None:
        src=setmatrix(src,matrix)
    if transfer is not None:
        src=settransfer(src,transfer)
    if primaries is not None:
        src=setprimaries(src,primaries)
    if chromaloc is not None:
        src=setchromaloc(src,chromaloc)
    return src


########################################################
########## HERE STARTS THE COPY-PASTE SECTION ##########
########################################################

#copy-paste from xyx98's xvs with some modification
#new feature
#radius2: similar to "sigma2"
#iterates: outputs all bm3d passes as a list in the order they take place
#iref: initial ref
#keepfloat: like what it said
#vt: v-bm3d type, 0 for good old bm3dcuda+bm3d.VAggregate, 1 for bm3dcuda.BM3Dv2
def bm3d(clip:vs.VideoNode,iref=None,sigma=[3,3,3],sigma2=None,preset="fast",preset2=None,mode=bm3d_mode_default,radius=0,radius2=None,chroma=False,fast=True,
            block_step1=None,bm_range1=None, ps_num1=None, ps_range1=None,
            block_step2=None,bm_range2=None, ps_num2=None, ps_range2=None,
            extractor_exp=bm3d_extractor_exp_default,device_id=0,bm_error_s="SSD",transform_2d_s="DCT",transform_1d_s="DCT",
            refine=1,dmode=0,iterates=False,keepfloat=False,vt=0):
    bits=clip.format.bits_per_sample
    clip=core.fmtc.bitdepth(clip,bits=32)
    iref=core.fmtc.bitdepth(iref,bits=32) if isinstance(iref,vs.VideoNode) else None
    if chroma is True and clip.format.id !=vs.YUV444PS:
        raise ValueError("chroma=True only works on yuv444")

    if radius2 is None:
        radius2=radius
    isvbm3d=radius+radius2>0

    if sigma2 is None:
        sigma2=sigma

    if preset2 is None:
        preset2=preset

    if preset not in ["fast","fastm","lc","lcm","np","npm","high","highm"] or preset2 not in ["fast","fastm","lc","lcm","np","npm","high","highm"]:
        raise ValueError("preset and preset2 must be 'fast','fastm','lc','lcm','np','npm','highm',or'high'")

    parmas1={
        #block_step,bm_range, ps_num, ps_range
        "fast":[8,9,2,4],
        "fastm":[7,9,2,4],
        "lc"  :[6,9,2,4],
        "lcm"  :[5,9,2,4],
        "np"  :[4,16,2,5],
        "npm"  :[3,16,2,5],
        "high":[3,16,2,7],
        "highm":[2,16,2,7],
    }

    vparmas1={
        #block_step,bm_range, ps_num, ps_range
        "fast":[8,7,2,4],
        "fastm":[7,7,2,4],
        "lc"  :[6,9,2,4],
        "lcm"  :[5,9,2,4],
        "np"  :[4,12,2,5],
        "npm"  :[3,12,2,5],
        "high":[3,16,2,7],
        "highm":[2,16,2,7],
    }

    parmas2={
        #block_step,bm_range, ps_num, ps_range
        "fast":[7,9,2,5],
        "fastm":[8,9,2,5],
        "lc"  :[5,9,2,5],
        "lcm"  :[6,9,2,5],
        "np"  :[3,16,2,6],
        "npm"  :[4,16,2,6],
        "high":[2,16,2,8],
        "highm":[3,16,2,8],
    }

    vparmas2={
        #block_step,bm_range, ps_num, ps_range
        "fast":[7,7,2,5],
        "fastm":[8,7,2,5],
        "lc"  :[5,9,2,5],
        "lcm"  :[6,9,2,5],
        "np"  :[3,12,2,6],
        "npm"  :[4,12,2,6],
        "high":[2,16,2,8],
        "highm":[3,16,2,8],
    }


    if isvbm3d:
        p1,p2=vparmas1,vparmas2
    else:
        p1,p2=parmas1,parmas2


    block_step1=p1[preset][0] if block_step1 is None else block_step1
    bm_range1=p1[preset][1] if bm_range1 is None else bm_range1
    ps_num1=p1[preset][2] if ps_num1 is None else ps_num1
    ps_range1=p1[preset][3] if ps_range1 is None else ps_range1

    block_step2=p2[preset2][0] if block_step2 is None else block_step2
    bm_range2=p2[preset2][1] if bm_range2 is None else bm_range2
    ps_num2=p2[preset2][2] if ps_num2 is None else ps_num2
    ps_range2=p2[preset2][3] if ps_range2 is None else ps_range2

    if iterates:
        outputs=list()
    if isvbm3d:
        flt=bm3d_core(clip,ref=iref,mode=mode,sigma=sigma,radius=radius,block_step=block_step1,bm_range=bm_range1,ps_num=ps_num1,ps_range=ps_range1,chroma=chroma,fast=fast,extractor_exp=extractor_exp,device_id=device_id,bm_error_s=bm_error_s,transform_2d_s=transform_2d_s,transform_1d_s=transform_1d_s,vt=vt)
        if radius>0 and vt==0:
            flt=core.bm3d.VAggregate(flt,radius=radius,sample=1)
        if iterates:
            outputs.append(core.fmtc.bitdepth(flt,bits=bits,dmode=dmode) if not keepfloat else flt)

        for i in range(refine):
            flt=bm3d_core(clip,ref=flt,mode=mode,sigma=sigma2,radius=radius2,block_step=block_step2,bm_range=bm_range2,ps_num=ps_num2,ps_range=ps_range2,chroma=chroma,fast=fast,extractor_exp=extractor_exp,device_id=device_id,bm_error_s=bm_error_s,transform_2d_s=transform_2d_s,transform_1d_s=transform_1d_s,vt=vt)
            if radius2>0 and vt==0:
                flt=core.bm3d.VAggregate(flt,radius=radius2,sample=1)
            if iterates:
                outputs.append(core.fmtc.bitdepth(flt,bits=bits,dmode=dmode) if not keepfloat else flt)

    else:
        flt=bm3d_core(clip,ref=iref,mode=mode,sigma=sigma,radius=radius,block_step=block_step1,bm_range=bm_range1,ps_num=ps_num1,ps_range=ps_range1,chroma=chroma,fast=fast,extractor_exp=extractor_exp,device_id=device_id,bm_error_s=bm_error_s,transform_2d_s=transform_2d_s,transform_1d_s=transform_1d_s,vt=vt)
        if iterates:
            outputs.append(core.fmtc.bitdepth(flt,bits=bits,dmode=dmode) if not keepfloat else flt)

        for i in range(refine):
            flt=bm3d_core(clip,ref=flt,mode=mode,sigma=sigma2,radius=radius2,block_step=block_step2,bm_range=bm_range2,ps_num=ps_num2,ps_range=ps_range2,chroma=chroma,fast=fast,extractor_exp=extractor_exp,device_id=device_id,bm_error_s=bm_error_s,transform_2d_s=transform_2d_s,transform_1d_s=transform_1d_s,vt=vt)
            if iterates:
                outputs.append(core.fmtc.bitdepth(flt,bits=bits,dmode=dmode) if not keepfloat else flt)

    if iterates:
        return outputs
    elif keepfloat:
        return flt
    else:
        return core.fmtc.bitdepth(flt,bits=bits,dmode=dmode)

#copy-paste from xyx98's xvs
def bm3d_core(clip,ref=None,mode="cpu",sigma=3.0,block_step=8,bm_range=9,radius=0,ps_num=2,ps_range=4,chroma=False,fast=True,extractor_exp=bm3d_extractor_exp_default,device_id=0,bm_error_s="SSD",transform_2d_s="DCT",transform_1d_s="DCT",vt=0):
    if mode not in ["cpu","cuda","cuda_rtc"]:
        raise ValueError("mode must be cpu,or cuda,or cuda_rtc")
    elif mode=="cpu":
        if vt==1:
            return core.bm3dcpu.BM3Dv2(clip,ref=ref,sigma=sigma,block_step=block_step,bm_range=bm_range,radius=radius,ps_num=ps_num,ps_range=ps_range,chroma=chroma)
        return core.bm3dcpu.BM3D(clip,ref=ref,sigma=sigma,block_step=block_step,bm_range=bm_range,radius=radius,ps_num=ps_num,ps_range=ps_range,chroma=chroma)
    elif mode=="cuda":
        if vt==1:
            return core.bm3dcuda.BM3Dv2(clip,ref=ref,sigma=sigma,block_step=block_step,bm_range=bm_range,radius=radius,ps_num=ps_num,ps_range=ps_range,chroma=chroma,fast=fast,extractor_exp=extractor_exp,device_id=device_id)
        return core.bm3dcuda.BM3D(clip,ref=ref,sigma=sigma,block_step=block_step,bm_range=bm_range,radius=radius,ps_num=ps_num,ps_range=ps_range,chroma=chroma,fast=fast,extractor_exp=extractor_exp,device_id=device_id)
    else:
        if vt==1:
            return core.bm3dcuda_rtc.BM3Dv2(clip,ref=ref,sigma=sigma,block_step=block_step,bm_range=bm_range,radius=radius,ps_num=ps_num,ps_range=ps_range,chroma=chroma,fast=fast,extractor_exp=extractor_exp,device_id=device_id,bm_error_s=bm_error_s,transform_2d_s=transform_2d_s,transform_1d_s=transform_1d_s)
        return core.bm3dcuda_rtc.BM3D(clip,ref=ref,sigma=sigma,block_step=block_step,bm_range=bm_range,radius=radius,ps_num=ps_num,ps_range=ps_range,chroma=chroma,fast=fast,extractor_exp=extractor_exp,device_id=device_id,bm_error_s=bm_error_s,transform_2d_s=transform_2d_s,transform_1d_s=transform_1d_s)

#copy-paste from xyx98's xvs with some modification
#new feature
#mask_gen_clip: an alternative clip can be provided for diff mask generation
#mask_operate_func: a function can be specified for mask operations after generation (e.g. expand, inpand and more)
#linear, sigmoid: do descale in linear or sigmoid light
def rescale(src:vs.VideoNode,kernel:str,w=None,h=None,mask=True,mask_dif_pix=2,show="result",postfilter_descaled=None,mthr:list[int]=[2,2],mask_gen_clip=None,mask_operate_func=None,linear=False,sigmoid=False,custom_nnedi3down=False,**args):
    if src.format.color_family not in [vs.YUV,vs.GRAY]:
        raise ValueError("input clip should be YUV or GRAY!")

    src_h,src_w=src.height,src.width
    if w is None and h is None:
        w,h=1280,720
    elif w is None:
        w=int(h*src_w/src_h)
    else:
        h=int(w*src_h/src_w)

    if w>=src_w or h>=src_h:
        raise ValueError("w,h should less than input resolution")

    kernel=kernel.strip().capitalize()
    if kernel not in ["Debilinear","Debicubic","Delanczos","Despline16","Despline36","Despline64"]:
        raise ValueError("unsupport kernel")

    src=core.fmtc.bitdepth(src,bits=16)
    luma=xvs.getY(src)
    tin='1886' if args.get("tin") is None else args.get("tin")
    fulls=False if args.get("fulls") is None else args.get("fulls")
    fulld=True if args.get("fulld") is None else args.get("fulld")
    if isinstance(mask_gen_clip,vs.VideoNode):
        luma=core.std.Interleave([luma,xvs.getY(mask_gen_clip)])
    if sigmoid:
        luma=core.fmtc.transfer(luma,transs=tin,transd='sigmoid',fulls=fulls,fulld=fulld)
    elif linear:
        luma=core.fmtc.transfer(luma,transs=tin,transd='linear',fulls=fulls,fulld=fulld)
    ####
    if kernel in ["Debilinear","Despline16","Despline36","Despline64"]:
        luma_de=eval("core.descale.{k}(luma.fmtc.bitdepth(bits=32),w,h)".format(k=kernel))
        luma_up=eval("core.resize.{k}(luma_de,src_w,src_h)".format(k=kernel[2:].capitalize())).fmtc.bitdepth(bits=16,dmode=1)
    elif kernel=="Debicubic":
        luma_de=core.descale.Debicubic(luma.fmtc.bitdepth(bits=32),w,h,b=args.get("b"),c=args.get("c"))
        luma_up=core.resize.Bicubic(luma_de,src_w,src_h,filter_param_a=args.get("b"),filter_param_b=args.get("c")).fmtc.bitdepth(bits=16,dmode=1)
    else:
        luma_de=core.descale.Delanczos(luma.fmtc.bitdepth(bits=32),w,h,taps=args.get("taps"))
        luma_up=core.resize.Lanczos(luma_de,src_w,src_h,filter_param_a=args.get("taps")).fmtc.bitdepth(bits=16,dmode=1)

    if isinstance(mask_gen_clip,vs.VideoNode):
        mclip=xvs.getY(mask_gen_clip)
        mclip_up=luma_up[1::2]
        luma=luma[::2]
        luma_de=luma_de[::2]
        luma_up=luma_up[::2]

    if postfilter_descaled is None:
        pass
    elif callable(postfilter_descaled):
        luma_de=postfilter_descaled(luma_de)
    else:
        raise ValueError("postfilter_descaled must be a function")

    nsize=3 if args.get("nsize") is None else args.get("nsize")#keep behavior before
    nns=args.get("nns")
    qual=2 if args.get("qual") is None else args.get("qual")#keep behavior before
    etype=args.get("etype")
    pscrn=1 if args.get("pscrn") is None else args.get("pscrn")
    exp=args.get("exp")
    mode=nnrs_mode_default if args.get("mode") is None else args.get("mode")

    if sigmoid:
        luma_de=core.fmtc.transfer(luma_de.fmtc.bitdepth(bits=16),transs='sigmoid',transd=tin,fulls=fulld,fulld=fulls)
    elif linear:
        luma_de=core.fmtc.transfer(luma_de.fmtc.bitdepth(bits=16),transs='linear',transd=tin,fulls=fulld,fulld=fulls)
    if callable(custom_nnedi3down):
        luma_rescale=nnrs.nnedi3_resample(luma_de,luma_de.width*2,luma_de.height*2,nsize=nsize,nns=nns,qual=qual,etype=etype,pscrn=pscrn,exp=exp,mode=mode)
        luma_rescale=custom_nnedi3down(luma_rescale,src_w,src_h).fmtc.bitdepth(bits=16)
    else:
        luma_rescale=nnrs.nnedi3_resample(luma_de,src_w,src_h,nsize=nsize,nns=nns,qual=qual,etype=etype,pscrn=pscrn,exp=exp,mode=mode).fmtc.bitdepth(bits=16)

    if mask:
        if not isinstance(mask_gen_clip,vs.VideoNode):
            mclip,mclip_up=luma,luma_up
        mask=core.std.Expr([mclip,mclip_up],"x y - abs").std.Binarize(mask_dif_pix*256)
        if callable(mask_operate_func):
            mask=mask_operate_func(mask)
        else:
            mask=xvs.expand(mask,cycle=mthr[0])
            mask=xvs.inpand(mask,cycle=mthr[1])

        luma_rescale=core.std.MaskedMerge(luma_rescale,xvs.getY(src),mask)

    if show=="descale":
        return luma_de
    elif show=="mask":
        return mask
    elif show=="both":
        return luma_de,mask

    if src.format.color_family==vs.GRAY:
        return luma_rescale
    else:
        return core.std.ShufflePlanes([luma_rescale,src],[0,1,2],vs.YUV)

#copy-paste from xyx98's xvs with some modification
def rescalef(src: vs.VideoNode,kernel: str,w=None,h=None,bh=None,bw=None,mask=True,mask_dif_pix=2,show="result",postfilter_descaled=None,selective=False,upper=0.0001,lower=0.00001,mthr:list[int]=[2,2],mask_gen_clip=None,mask_operate_func=None,linear=False,sigmoid=False,custom_nnedi3down=False,**args):
    #for decimal resolution descale,refer to GetFnative
    if src.format.color_family not in [vs.YUV,vs.GRAY]:
        raise ValueError("input clip should be YUV or GRAY!")

    src_h,src_w=src.height,src.width
    if w is None and h is None:
        w,h=1280,720
    elif w is None:
        w=int(h*src_w/src_h)
    else:
        h=int(w*src_h/src_w)

    if bh is None:
        bh=1080

    if w>=src_w or h>=src_h:
        raise ValueError("w,h should less than input resolution")

    kernel=kernel.strip().capitalize()
    if kernel not in ["Debilinear","Debicubic","Delanczos","Despline16","Despline36","Despline64"]:
        raise ValueError("unsupport kernel")

    src=core.fmtc.bitdepth(src,bits=16)
    luma=xvs.getY(src)
    tin='1886' if args.get("tin") is None else args.get("tin")
    fulls=False if args.get("fulls") is None else args.get("fulls")
    fulld=True if args.get("fulld") is None else args.get("fulld")
    if isinstance(mask_gen_clip,vs.VideoNode):
        luma=core.std.Interleave([luma,xvs.getY(mask_gen_clip)])
    if sigmoid:
        luma=core.fmtc.transfer(luma,transs=tin,transd='sigmoid',fulls=fulls,fulld=fulld)
    elif linear:
        luma=core.fmtc.transfer(luma,transs=tin,transd='linear',fulls=fulls,fulld=fulld)
    cargs=xvs.cropping_args(src.width,src.height,h,bh,bw)
    ####
    if kernel in ["Debilinear","Despline16","Despline36","Despline64"]:
        luma_de=eval("core.descale.{k}(luma.fmtc.bitdepth(bits=32),**cargs.descale_gen())".format(k=kernel))
        luma_up=eval("core.resize.{k}(luma_de,**cargs.resize_gen())".format(k=kernel[2:].capitalize()))
    elif kernel=="Debicubic":
        luma_de=core.descale.Debicubic(luma.fmtc.bitdepth(bits=32),b=args.get("b"),c=args.get("c"),**cargs.descale_gen())
        luma_up=core.resize.Bicubic(luma_de,filter_param_a=args.get("b"),filter_param_b=args.get("c"),**cargs.resize_gen())
    else:
        luma_de=core.descale.Delanczos(luma.fmtc.bitdepth(bits=32),taps=args.get("taps"),**cargs.descale_gen())
        luma_up=core.resize.Lanczos(luma_de,filter_param_a=args.get("taps"),**cargs.resize_gen())#

    if isinstance(mask_gen_clip,vs.VideoNode):
        mclip=xvs.getY(mask_gen_clip)
        mclip_up=luma_up[1::2]
        luma=luma[::2]
        luma_de=luma_de[::2]
        luma_up=luma_up[::2]

    diff = core.std.Expr([luma.fmtc.bitdepth(bits=32), luma_up], f'x y - abs dup 0.015 > swap 0 ?').std.Crop(10, 10, 10, 10).std.PlaneStats()

    if postfilter_descaled is None:
        pass
    elif callable(postfilter_descaled):
        luma_de=postfilter_descaled(luma_de)
    else:
        raise ValueError("postfilter_descaled must be a function")

    nsize=3 if args.get("nsize") is None else args.get("nsize")#keep behavior before
    nns=args.get("nns")
    qual=2 if args.get("qual") is None else args.get("qual")#keep behavior before
    etype=args.get("etype")
    pscrn=1 if args.get("pscrn") is None else args.get("pscrn")
    exp=args.get("exp")
    mode=nnrs_mode_default if args.get("mode") is None else args.get("mode")

    if sigmoid:
        luma_de=core.fmtc.transfer(luma_de.fmtc.bitdepth(bits=16),transs='sigmoid',transd=tin,fulls=fulld,fulld=fulls)
    elif linear:
        luma_de=core.fmtc.transfer(luma_de.fmtc.bitdepth(bits=16),transs='linear',transd=tin,fulls=fulld,fulld=fulls)
    if callable(custom_nnedi3down):
        _cargs=cargs.nnrs_gen()
        del _cargs['target_width'],_cargs['target_height']
        luma_rescale=nnrs.nnedi3_resample(luma_de,luma_de.width*2,luma_de.height*2,nsize=nsize,nns=nns,qual=qual,etype=etype,pscrn=pscrn,exp=exp,mode=mode,**_cargs)
        luma_rescale=custom_nnedi3down(luma_rescale,src_w,src_h).fmtc.bitdepth(bits=16)
    else:
        luma_rescale=nnrs.nnedi3_resample(luma_de,nsize=nsize,nns=nns,qual=qual,etype=etype,pscrn=pscrn,exp=exp,mode=mode,**cargs.nnrs_gen()).fmtc.bitdepth(bits=16)

    def calc(n,f):
        fout=f[1].copy()
        fout.props["diff"]=f[0].props["PlaneStatsAverage"]
        return fout

    luma_rescale=core.std.ModifyFrame(luma_rescale,[diff,luma_rescale],calc)

    if mask:
        if not isinstance(mask_gen_clip,vs.VideoNode):
            mclip,mclip_up=luma,luma_up
        mask=core.std.Expr([mclip,mclip_up.fmtc.bitdepth(bits=16,dmode=1)],"x y - abs").std.Binarize(mask_dif_pix*256)
        if callable(mask_operate_func):
            mask=mask_operate_func(mask)
        else:
            mask=xvs.expand(mask,cycle=mthr[0])
            mask=xvs.inpand(mask,cycle=mthr[1])

        luma_rescale=core.std.MaskedMerge(luma_rescale,xvs.getY(src),mask)

    if selective:
        base=upper-lower
        #x:rescale y:src
        expr=f"x.diff {upper} > y x.diff {lower} < x {upper} x.diff -  {base} / y * x.diff {lower} - {base} / x * + ? ?"
        luma_rescale=core.akarin.Expr([luma_rescale,luma], expr)

    if show=="descale":
        return luma_de
    elif show=="mask":
        return mask
    elif show=="both":
        return luma_de,mask
    elif show=="diff":
        return core.text.FrameProps(luma_rescale,"diff", scale=2)

    if src.format.color_family==vs.GRAY:
        return luma_rescale
    else:
        return core.std.ShufflePlanes([luma_rescale,src],[0,1,2],vs.YUV)

#copy-paste from xyx98's xvs with some modification
def multirescale(clip:vs.VideoNode,kernels:list[dict],w:Optional[int]=None,h:Optional[int]=None,mask:bool=True,mask_dif_pix:float=2.5,postfilter_descaled=None,selective_disable:bool=False,disable_thr:float=0.00001,showinfo=False,mthr:list[int]=[2,2],mask_gen_clip=None,mask_operate_func=None,linear=False,sigmoid=False,tin='1886',fulls=False,fulld=True,**args):
    clip=core.fmtc.bitdepth(clip,bits=16)
    luma=xvs.getY(clip)
    src_h,src_w=clip.height,clip.width
    def getwh(w,h):
        if w is None and h is None:
            w,h=1280,720
        elif w is None:
            w=int(h*src_w/src_h)
        elif h is None:
            h=int(w*src_h/src_w)

        if w>=src_w or h>=src_h:
            raise ValueError("w,h should less than input resolution")
        return w,h

    w,h=getwh(w,h)

    info_gobal=f"gobal:\nresolution:{w}x{h}\tmask:{mask}\tmask_dif_pix:{mask_dif_pix}\tpostfilter_descaled:{'yes' if postfilter_descaled else 'no'}\nselective_disable:{selective_disable}\tdisable_thr:{disable_thr:f}\nextra:{str(args)}"
    rescales=[]
    total=len(kernels)
    for i in kernels:
        fmode=False if i.get("fmode") is None else i.get("fmode")
        k=i["k"][2:]
        kb,kc,ktaps=i.get("b"),i.get("c"),i.get("taps")
        kw,kh=i.get("w"),i.get("h")
        if kw is not None or kh is not None:
            kw,kh=getwh(kw,kh)
        else:
            kw,kh=w,h
        kmask=mask if i.get("mask") is None else i.get("mask")
        kmdp=mask_dif_pix if i.get("mask_dif_pix") is None else i.get("mask_dif_pix")
        kpp=postfilter_descaled if i.get("postfilter_descaled") is None else i.get("postfilter_descaled")
        multiple=1 if i.get("multiple") is None else i.get("multiple")
        kmthr=mthr if i.get("mthr") is None else i.get("mthr")
        mgc=mask_gen_clip if i.get("mask_gen_clip") is None else i.get("mask_gen_clip")
        mof=mask_operate_func if i.get("mask_operate_func") is None else i.get("mask_operate_func")
        lin=linear if i.get("linear") is None else i.get("linear")
        sig=sigmoid if i.get("sigmoid") is None else i.get("sigmoid")
        tin=tin if i.get("tin") is None else i.get("tin")
        fulls=fulls if i.get("fulls") is None else i.get("fulls")
        fulld=fulld if i.get("fulld") is None else i.get("fulld")

        if not fmode:
            rescales.append(MRcore(luma,kernel=k,w=kw,h=kh,mask=kmask,mask_dif_pix=kmdp,postfilter_descaled=kpp,taps=ktaps,b=kb,c=kc,multiple=multiple,mthr=kmthr,mask_gen_clip=mgc,mask_operate_func=mof,linear=lin,sigmoid=sig,**args))

        else:
            kbh=src_h if i.get("bh") is None else i.get("bh")
            kbw=i.get("bw")
            rescales.append(MRcoref(luma,kernel=k,w=kw,h=kh,bh=kbh,bw=kbw,mask=kmask,mask_dif_pix=kmdp,postfilter_descaled=kpp,mthr=kmthr,taps=ktaps,b=kb,c=kc,multiple=multiple,maskpp=mof,**args))


    def selector(n,f,src,clips):
        kernels_info=[]
        index,mindiff=0,f[0].props["diff"]
        for i in range(total):
            tmpdiff=f[i].props["diff"]
            kernels_info.append(f"kernel {i}:\t{kernels[i]}\n{tmpdiff:.10f}")
            if tmpdiff<mindiff:
                index,mindiff=i,tmpdiff

        info=info_gobal+"\n--------------------\n"+("\n--------------------\n").join(kernels_info)+"\n--------------------\ncurrent usage:\n"
        if selective_disable and mindiff>disable_thr:
            last=src
            info+="source"
        else:
            last=clips[index]
            info+=kernels_info[index]
        if showinfo:
            last=core.text.Text(last,info.replace("\t","    "))
        return last.std.SetFrameProp('kindex',intval=index)

    last=core.std.FrameEval(luma,partial(selector,src=luma,clips=rescales),prop_src=rescales)
    if clip.format.color_family==vs.GRAY:
        return last
    else:
        return core.std.ShufflePlanes([last,clip],[0,1,2],vs.YUV)

#copy-paste from xyx98's xvs with some modification
def MRcore(clip:vs.VideoNode,kernel:str,w:int,h:int,mask: Union[bool,vs.VideoNode]=True,mask_dif_pix:float=2,postfilter_descaled=None,taps:int=3,b:float=0,c:float=0.5,multiple:float=1,mthr:list[int]=[2,2],mask_gen_clip=None,mask_operate_func=None,linear=False,sigmoid=False,tin='1886',fulls=False,fulld=True,custom_nnedi3down=False,show:str='result',**args):
    src_w,src_h=clip.width,clip.height
    clipo=clip
    if isinstance(mask_gen_clip,vs.VideoNode):
        clip32=core.std.Interleave([clip32,xvs.getY(mask_gen_clip)])
    if sigmoid:
        clip=core.fmtc.transfer(clip,transs=tin,transd='sigmoid',fulls=fulls,fulld=fulld)
    elif linear:
        clip=core.fmtc.transfer(clip,transs=tin,transd='linear',fulls=fulls,fulld=fulld)
    clip32=core.fmtc.bitdepth(clip,bits=32)
    descaled=core.descale.Descale(clip32,width=w,height=h,kernel=kernel.lower(),taps=taps,b=b,c=c)
    upscaled=xvs.resize_core(kernel.capitalize(),taps,b,c)(descaled,src_w,src_h)
    if isinstance(mask_gen_clip,vs.VideoNode):
        mclip=xvs.getY(mask_gen_clip)
        mclip_up=upscaled[1::2]
        clip32=clip32[::2]
        descaled=descaled[::2]
        upscaled=upscaled[::2]
    diff=core.std.Expr([clip32,upscaled],"x y - abs dup 0.015 > swap 0 ?").std.PlaneStats()

    def calc(n,f):
        fout=f[1].copy()
        fout.props["diff"]=f[0].props["PlaneStatsAverage"]*multiple
        return fout

    if postfilter_descaled is None:
        pass
    elif callable(postfilter_descaled):
        descaled=postfilter_descaled(descaled)
    else:
        raise ValueError("postfilter_descaled must be a function")

    nsize=3 if args.get("nsize") is None else args.get("nsize")
    nns=args.get("nns")
    qual=2 if args.get("qual") is None else args.get("qual")
    etype=args.get("etype")
    pscrn=1 if args.get("pscrn") is None else args.get("pscrn")
    exp=args.get("exp")
    mode=nnrs_mode_default if args.get("mode") is None else args.get("mode")
    if mode=='eval':
        return core.std.ModifyFrame(clipo,[diff,clipo],calc)
    if sigmoid:
        descaled=core.fmtc.transfer(descaled.fmtc.bitdepth(bits=16),transs='sigmoid',transd=tin,fulls=fulld,fulld=fulls)
    elif linear:
        descaled=core.fmtc.transfer(descaled.fmtc.bitdepth(bits=16),transs='linear',transd=tin,fulls=fulld,fulld=fulls)
    if callable(custom_nnedi3down):
        rescale=nnrs.nnedi3_resample(descaled,descaled.width*2,descaled.height*2,nsize=nsize,nns=nns,qual=qual,etype=etype,pscrn=pscrn,exp=exp,mode=mode)
        rescale=custom_nnedi3down(rescale,src_w,src_h).fmtc.bitdepth(bits=16)
    else:
        rescale=nnrs.nnedi3_resample(descaled,src_w,src_h,nsize=nsize,nns=nns,qual=qual,etype=etype,pscrn=pscrn,exp=exp,mode=mode).fmtc.bitdepth(bits=16)

    if mask is True:
        if not isinstance(mask_gen_clip,vs.VideoNode):
            mclip=clip
            mclip_up=upscaled
        mask=core.std.Expr([mclip,mclip_up.fmtc.bitdepth(bits=16,dmode=1)],"x y - abs").std.Binarize(mask_dif_pix*256)
        if callable(mask_operate_func):
            mask=mask_operate_func(mask)
        else:
            mask=xvs.expand(mask,cycle=mthr[0])
            mask=xvs.inpand(mask,cycle=mthr[1])
        rescale=core.std.MaskedMerge(rescale,clipo,mask)
    elif isinstance(mask,vs.VideoNode):
        if mask.width!=src_w or mask.height!=src_h or mask.format.color_family!=vs.GRAY:
            raise ValueError("mask should have same resolution as source,and should be GRAY")
        mask=core.fmtc.bitdepth(mask,bits=16,dmode=1)
        rescale=core.std.MaskedMerge(rescale,clip,mask)
    else:
        mask=core.std.BlankClip(rescale)

    if show.lower()=="result":
        return core.std.ModifyFrame(rescale,[diff,rescale],calc)
    elif show.lower()=="mask" and mask:
        return core.std.ModifyFrame(mask,[diff,mask],calc)
    elif show.lower()=="descale":
        return descaled #after postfilter_descaled
    elif show.lower()=="both": #result,mask,descaled
        return core.std.ModifyFrame(rescale,[diff,rescale],calc),core.std.ModifyFrame(mask,[diff,mask],calc),descaled

#copy-paste from xyx98's xvs with some modification
def MRcoref(clip:vs.VideoNode,kernel:str,w:float,h:float,bh:int,bw:int=None,mask: Union[bool,vs.VideoNode]=True,mask_dif_pix:float=2,postfilter_descaled=None,mthr:list[int]=[2,2],taps:int=3,b:float=0,c:float=0.5,multiple:float=1,maskpp=None,show:str="result",mask_gen_clip=None,linear=False,sigmoid=False,tin='1886',fulls=False,fulld=True,custom_nnedi3down=False,**args):

    src_w,src_h=clip.width,clip.height
    clipo=clip
    if isinstance(mask_gen_clip,vs.VideoNode):
        clip32=core.std.Interleave([clip32,xvs.getY(mask_gen_clip)])
    if sigmoid:
        clip=core.fmtc.transfer(clip,transs=tin,transd='sigmoid',fulls=fulls,fulld=fulld)
    elif linear:
        clip=core.fmtc.transfer(clip,transs=tin,transd='linear',fulls=fulls,fulld=fulld)
    cargs=xvs.cropping_args(src_w,src_h,h,bh,bw)
    clip32=core.fmtc.bitdepth(clip,bits=32)
    descaled=core.descale.Descale(clip32,kernel=kernel.lower(),taps=taps,b=b,c=c,**cargs.descale_gen())
    upscaled=xvs.resize_core(kernel.capitalize(),taps,b,c)(descaled,**cargs.resize_gen())
    if isinstance(mask_gen_clip,vs.VideoNode):
        mclip=xvs.getY(mask_gen_clip)
        mclip_up=upscaled[1::2]
        clip32=clip32[::2]
        descaled=descaled[::2]
        upscaled=upscaled[::2]
    diff=core.std.Expr([clip32,upscaled],"x y - abs dup 0.015 > swap 0 ?").std.Crop(10, 10, 10, 10).std.PlaneStats()
    def calc(n,f):
        fout=f[1].copy()
        fout.props["diff"]=f[0].props["PlaneStatsAverage"]*multiple
        return fout

    if postfilter_descaled is None:
        pass
    elif callable(postfilter_descaled):
        descaled=postfilter_descaled(descaled)
    else:
        raise ValueError("postfilter_descaled must be a function")

    nsize=3 if args.get("nsize") is None else args.get("nsize")
    nns=args.get("nns")
    qual=2 if args.get("qual") is None else args.get("qual")
    etype=args.get("etype")
    pscrn=1 if args.get("pscrn") is None else args.get("pscrn")
    exp=args.get("exp")
    mode=nnrs_mode_default if args.get("mode") is None else args.get("mode")
    if mode=='eval':
        return core.std.ModifyFrame(clipo,[diff,clipo],calc)
    if sigmoid:
        descaled=core.fmtc.transfer(descaled.fmtc.bitdepth(bits=16),transs='sigmoid',transd=tin,fulls=fulld,fulld=fulls)
    elif linear:
        descaled=core.fmtc.transfer(descaled.fmtc.bitdepth(bits=16),transs='linear',transd=tin,fulls=fulld,fulld=fulls)

    if callable(custom_nnedi3down):
        _cargs=cargs.nnrs_gen()
        del _cargs['target_width'],_cargs['target_height']
        rescale=nnrs.nnedi3_resample(descaled,descaled.width*2,descaled.height*2,nsize=nsize,nns=nns,qual=qual,etype=etype,pscrn=pscrn,exp=exp,mode=mode,**_cargs)
        rescale=custom_nnedi3down(rescale,src_w,src_h).fmtc.bitdepth(bits=16)
    else:
        rescale=nnrs.nnedi3_resample(descaled,nsize=nsize,nns=nns,qual=qual,etype=etype,pscrn=pscrn,exp=exp,mode=mode,**cargs.nnrs_gen()).fmtc.bitdepth(bits=16)

    if mask is True:
        if not isinstance(mask_gen_clip,vs.VideoNode):
            mclip=clip
            mclip_up=upscaled
        mask=core.std.Expr([mclip,mclip_up.fmtc.bitdepth(bits=16,dmode=1)],"x y - abs").std.Binarize(mask_dif_pix*256)
        if callable(maskpp):
            mask=maskpp(mask)
        else:
            mask=xvs.expand(mask,cycle=mthr[0])
            mask=xvs.inpand(mask,cycle=mthr[1])
        rescale=core.std.MaskedMerge(rescale,clip,mask)
    elif isinstance(mask,vs.VideoNode):
        if mask.width!=src_w or mask.height!=src_h or mask.format.color_family!=vs.GRAY:
            raise ValueError("mask should have same resolution as source,and should be GRAY")
        mask=core.fmtc.bitdepth(mask,bits=16,dmode=1)
        rescale=core.std.MaskedMerge(rescale,clip,mask)
    else:
        mask=core.std.BlankClip(rescale)

    if show.lower()=="result":
        return core.std.ModifyFrame(rescale,[diff,rescale],calc)
    elif show.lower()=="mask" and mask:
        return core.std.ModifyFrame(mask,[diff,mask],calc)
    elif show.lower()=="descale":
        return descaled #after postfilter_descaled
    elif show.lower()=="both": #result,mask,descaled
        return core.std.ModifyFrame(rescale,[diff,rescale],calc),core.std.ModifyFrame(mask,[diff,mask],calc),descaled


#copy-paste from HolyWu's havsfunc
def ContraSharpening(
    denoised: vs.VideoNode, original: vs.VideoNode, radius: int = 1, rep: int = 1, planes: Optional[Union[int, Sequence[int]]] = None
) -> vs.VideoNode:
    '''
    contra-sharpening: sharpen the denoised clip, but don't add more to any pixel than what was removed previously.

    Parameters:
        denoised: Denoised clip to sharpen.

        original: Original clip before denoising.

        radius: Spatial radius for contra-sharpening.

        rep: Mode of repair to limit the difference.

        planes: Specifies which planes will be processed. Any unprocessed planes will be simply copied.
            By default only luma plane will be processed for non-RGB formats.
    '''
    if not (isinstance(denoised, vs.VideoNode) and isinstance(original, vs.VideoNode)):
        raise vs.Error('ContraSharpening: this is not a clip')

    if denoised.format.id != original.format.id:
        raise vs.Error('ContraSharpening: clips must have the same format')

    neutral = 1 << (denoised.format.bits_per_sample - 1)

    plane_range = range(denoised.format.num_planes)

    if planes is None:
        planes = [0] if denoised.format.color_family != vs.RGB else [0, 1, 2]
    elif isinstance(planes, int):
        planes = [planes]

    pad = 2 if radius < 3 else 4
    denoised = rpclip(denoised, pad)
    original = rpclip(original, pad)

    matrix1 = [1, 2, 1, 2, 4, 2, 1, 2, 1]
    matrix2 = [1, 1, 1, 1, 1, 1, 1, 1, 1]

    # damp down remaining spots of the denoised clip
    s = MinBlur(denoised, radius, planes)
    # the difference achieved by the denoising
    allD = core.std.MakeDiff(original, denoised, planes=planes)

    RG11 = s.std.Convolution(matrix=matrix1, planes=planes)
    if radius >= 2:
        RG11 = RG11.std.Convolution(matrix=matrix2, planes=planes)
    if radius >= 3:
        RG11 = RG11.std.Convolution(matrix=matrix2, planes=planes)

    # the difference of a simple kernel blur
    ssD = core.std.MakeDiff(s, RG11, planes=planes)
    # limit the difference to the max of what the denoising removed locally
    ssDD = core.rgvs.Repair(ssD, allD, mode=[rep if i in planes else 0 for i in plane_range])
    # abs(diff) after limiting may not be bigger than before
    ssDD = core.std.Expr([ssDD, ssD], expr=[f'x {neutral} - abs y {neutral} - abs < x y ?' if i in planes else '' for i in plane_range])
    # apply the limited difference (sharpening is just inverse blurring)
    last = core.std.MergeDiff(denoised, ssDD, planes=planes)
    return last.std.Crop(pad, pad, pad, pad)

#copy-paste from HolyWu's havsfunc
def MinBlur(clp: vs.VideoNode, r: int = 1, planes: Optional[Union[int, Sequence[int]]] = None) -> vs.VideoNode:
    '''Nifty Gauss/Median combination'''
    from mvsfunc import LimitFilter

    if not isinstance(clp, vs.VideoNode):
        raise vs.Error('MinBlur: this is not a clip')

    plane_range = range(clp.format.num_planes)

    if planes is None:
        planes = list(plane_range)
    elif isinstance(planes, int):
        planes = [planes]

    matrix1 = [1, 2, 1, 2, 4, 2, 1, 2, 1]
    matrix2 = [1, 1, 1, 1, 1, 1, 1, 1, 1]

    if r <= 0:
        RG11 = sbr(clp, planes=planes)
        RG4 = clp.std.Median(planes=planes)
    elif r == 1:
        RG11 = clp.std.Convolution(matrix=matrix1, planes=planes)
        RG4 = clp.std.Median(planes=planes)
    elif r == 2:
        RG11 = clp.std.Convolution(matrix=matrix1, planes=planes).std.Convolution(matrix=matrix2, planes=planes)
        RG4 = clp.ctmf.CTMF(radius=2, planes=planes)
    else:
        RG11 = clp.std.Convolution(matrix=matrix1, planes=planes).std.Convolution(matrix=matrix2, planes=planes).std.Convolution(matrix=matrix2, planes=planes)
        if clp.format.bits_per_sample == 16:
            s16 = clp
            RG4 = core.fmtc.bitdepth(clp, bits=12, dmode=1).ctmf.CTMF(radius=3, planes=planes)
            RG4 = LimitFilter(s16, core.fmtc.bitdepth(RG4, bits=16), thr=0.0625, elast=2, planes=planes)
        else:
            RG4 = clp.ctmf.CTMF(radius=3, planes=planes)

    return core.std.Expr([clp, RG11, RG4], expr=['x y - x z - * 0 < x x y - abs x z - abs < y z ? ?' if i in planes else '' for i in plane_range])

#copy-paste from HolyWu's havsfunc
def sbr(c: vs.VideoNode, r: int = 1, planes: Optional[Union[int, Sequence[int]]] = None) -> vs.VideoNode:
    '''make a highpass on a blur's difference (well, kind of that)'''
    if not isinstance(c, vs.VideoNode):
        raise vs.Error('sbr: this is not a clip')

    neutral = 1 << (c.format.bits_per_sample - 1) if c.format.sample_type == vs.INTEGER else 0.0

    plane_range = range(c.format.num_planes)

    if planes is None:
        planes = list(plane_range)
    elif isinstance(planes, int):
        planes = [planes]

    matrix1 = [1, 2, 1, 2, 4, 2, 1, 2, 1]
    matrix2 = [1, 1, 1, 1, 1, 1, 1, 1, 1]

    RG11 = c.std.Convolution(matrix=matrix1, planes=planes)
    if r >= 2:
        RG11 = RG11.std.Convolution(matrix=matrix2, planes=planes)
    if r >= 3:
        RG11 = RG11.std.Convolution(matrix=matrix2, planes=planes)

    RG11D = core.std.MakeDiff(c, RG11, planes=planes)

    RG11DS = RG11D.std.Convolution(matrix=matrix1, planes=planes)
    if r >= 2:
        RG11DS = RG11DS.std.Convolution(matrix=matrix2, planes=planes)
    if r >= 3:
        RG11DS = RG11DS.std.Convolution(matrix=matrix2, planes=planes)

    RG11DD = core.std.Expr(
        [RG11D, RG11DS],
        expr=[f'x y - x {neutral} - * 0 < {neutral} x y - abs x {neutral} - abs < x y - {neutral} + x ? ?' if i in planes else '' for i in plane_range],
    )
    return core.std.MakeDiff(c, RG11DD, planes=planes)

#copy-paste from xyx98's xvs with some modification
def SCSharpen(clip:vs.VideoNode,ref:vs.VideoNode,max_sharpen_weight=3/7,min_sharpen_weight=0,casstr=0.7):
    """
    Sharpness Considered Sharpen:
    It mainly design for sharpen a bad source after blurry filtered such as strong AA, and source unsuited to be reference when you want sharpen filtered clip to match the sharpness of source.
    It use cas as sharpen core,and calculate sharpness of source(reference clip),filtered clip (the clip you want sharpen),and sharpen clip(by cas).Use these sharpness information adjust merge weight of filtered clip and sharpen clip.
    ############################
    *If clean is True,use haf.EdgeCleaner clean edge after sharpen.*
    arg clean is removed because of the deprecation of havsfunc in this script.
    Don't use high max_sharpen_weight or you might need addition filter to resolve artifacts cause by cas(1).
    only luma processed,output is always 16bit.
    """
    # trust the user( ͡° ͜ʖ ͡°) 爱来自中国
    # if max_sharpen_weight >1 or max_sharpen_weight <=0 :
    #     raise ValueError("max_sharpen_weight should in (0,1]")

    # if min_sharpen_weight >1 or min_sharpen_weight <0  or max_sharpen_weight<min_sharpen_weight:
    #     raise ValueError("min_sharpen_weight should in [0,1] and less than max_sharpen_weight")

    ref,clip=core.fmtc.bitdepth(ref,bits=16),core.fmtc.bitdepth(clip,bits=16)
    if clip.format.color_family == vs.YUV:
        isYUV=True
        last=xvs.getY(clip)
    elif clip.format.color_family == vs.GRAY:
        last=clip
        isYUV=False
    else:
        raise vs.ValueError("clip must be YUV or GRAY")

    if ref.format.color_family == vs.YUV:
        ref=xvs.getY(ref)
    elif ref.format.color_family == vs.GRAY:
        pass
    else:
        raise vs.ValueError("ref must be YUV or GRAY")

    sharp=core.cas.CAS(last,casstr,0)
    ref,last,sharp=map(getsharpness,[ref,last,sharp])
    #########################
    base=" z.sharpness y.sharpness - "
    k1=f"x.sharpness y.sharpness - {base} /"
    k2=f"z.sharpness x.sharpness - {base} /"
    L1=max_sharpen_weight
    L2=1-L1
    L3=min_sharpen_weight
    L4=1-L3

    expr=f"{base} 0 = {L1} z * {L2} y * + {k1} {L1} > {L1} z * {L2} y * + {k1} {L3} < {L3} z * {L4} y * + {k1} z * {k2} y * + ? ? ?"
    last=core.akarin.Expr([ref,last,sharp],expr)

    if isYUV:
        last=core.std.ShufflePlanes([last,clip],[0,1,2],vs.YUV)

    return last
#alias
#i hate uppercase func/arg name somehow
scsharpen=SCSharpen

#copy-paste from xyx98's xvs
def getsharpness(clip,show=False):

    def calc(n,f):
        fout=f[1].copy()
        fout.props["sharpness"]=f[0].props["PlaneStatsAverage"]*65535
        return fout

    luma=xvs.getY(clip).fmtc.bitdepth(bits=16)
    blur=core.rgvs.RemoveGrain(luma, 20)
    dif=core.akarin.Expr([luma,blur],[f"x y - 65535 / 2 pow 65535 *"])
    dif=core.std.PlaneStats(dif)
    last=core.std.ModifyFrame(clip,[dif,clip],calc)
    return core.text.FrameProps(last,"sharpness",scale=2) if show else last