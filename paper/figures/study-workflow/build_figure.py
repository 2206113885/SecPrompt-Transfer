# /// script
# requires-python = ">=3.10"
# dependencies = ["cairosvg==2.9.1", "matplotlib==3.11.2", "pymupdf==1.28.2", "PyYAML==6.0.3", "fonttools==4.68.0"]
# ///
"""Rebuild Figure 1 from the canonical YAML and editable native Draw.io cells.

Usage: uv run --isolated --no-project build_figure.py
Optional: --drawio-cli /path/to/drawio/scripts/cli.js --font-dir /path/to/Arial

The installed skill's SVG renderer leaves MathJax labels untypeset. This exporter
replaces each label with measured Arial text and vector STIX math, at identical
node coordinates. No raster layer or LaTeX/TikZ drawing is used.
"""
from __future__ import annotations
import argparse, base64, hashlib, importlib.metadata, io, json, os, re, subprocess, tempfile
from pathlib import Path
from xml.etree import ElementTree as ET

ROOT = Path(__file__).resolve().parent
BASENAME = 'Fig1-study-workflow'
SVGNS = 'http://www.w3.org/2000/svg'
ET.register_namespace('', SVGNS)

def tag(name): return '{'+SVGNS+'}'+name

def run(argv):
    r=subprocess.run([str(x) for x in argv],check=True,text=True,capture_output=True)
    return r.stdout + r.stderr

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--drawio-cli',type=Path,default=Path.home()/'.codex/skills/drawio/scripts/cli.js')
    ap.add_argument('--font-dir',type=Path,default=Path('/System/Library/Fonts/Supplemental'))
    args=ap.parse_args()
    os.environ.setdefault('DYLD_FALLBACK_LIBRARY_PATH','/opt/homebrew/lib:/usr/local/lib')
    import yaml, pymupdf, matplotlib
    matplotlib.use('Agg')
    matplotlib.rcParams['mathtext.fontset']='stix'
    from matplotlib.textpath import TextPath
    from matplotlib.font_manager import FontProperties
    from matplotlib.mathtext import MathTextParser
    from matplotlib.path import Path as MPath
    import cairosvg

    spec=yaml.safe_load((ROOT/(BASENAME+'.yaml')).read_text())
    canvas_w,canvas_h=map(float,spec['meta']['canvas'].split('x'))
    width_pt=float(spec['meta']['print']['widthPt'])
    height_pt=width_pt*canvas_h/canvas_w
    font_files={
        'regular':args.font_dir/'Arial.ttf', 'bold':args.font_dir/'Arial Bold.ttf',
        'italic':args.font_dir/'Arial Italic.ttf',
    }
    for f in font_files.values():
        if not f.exists(): raise FileNotFoundError(f'Arial font required: {f}')
    fonts={k:pymupdf.Font(fontfile=str(v)) for k,v in font_files.items()}
    math_parser=MathTextParser('path')

    def parts(label,fs,bold=False):
        runs=[]
        for i,part in enumerate(re.split(r'(\\\(.*?\\\))',label)):
            if not part: continue
            if part.startswith('\\('):
                tex='$'+part[2:-2]+'$'
                parsed=math_parser.parse(tex,dpi=72,prop=FontProperties(size=fs))
                path=TextPath((0,0),tex,size=fs,prop=FontProperties(family='STIXGeneral'))
                runs.append(('math',part,float(parsed.width),path))
            else:
                f=fonts['bold' if bold else 'regular']
                runs.append(('text',part,f.text_length(part,fontsize=fs),None))
        return runs

    def svg_path(path):
        out=[]
        for verts,code in path.iter_segments(curves=True,simplify=False):
            coords=' '.join(f'{v:.5f}' for v in verts)
            if code==MPath.MOVETO: out.append('M '+coords)
            elif code==MPath.LINETO: out.append('L '+coords)
            elif code==MPath.CURVE3: out.append('Q '+coords)
            elif code==MPath.CURVE4: out.append('C '+coords)
            elif code==MPath.CLOSEPOLY: out.append('Z')
        return ' '.join(out)

    # Record font metrics for every real label; backgrounds and stack sheets are
    # intentional containment, not node-overlap errors.
    label_metrics=[]
    for n in spec['nodes']:
        if not n['label'].strip(): continue
        s=n['style']; fs=s['fontSize']; bold=s.get('fontWeight',400)>=600
        lines=n['label'].split('\n'); widths=[sum(p[2] for p in parts(l,fs,bold)) for l in lines]
        b=n['bounds']
        label_metrics.append(dict(id=n['id'],label=n['label'],font_px=fs,font_pt=fs*width_pt/canvas_w,
            max_line_width=round(max(widths),2),available_width=b['width'],
            visual_height=round((len(lines)-1)*fs*1.4+fs,2),available_height=b['height'],
            horizontal_fits=max(widths)<=b['width']-2))

    with tempfile.TemporaryDirectory(prefix='emse-figure1-') as tmp:
        tmp=Path(tmp)
        log=run(['node',args.drawio_cli,ROOT/(BASENAME+'.yaml'),ROOT/(BASENAME+'.drawio'),'--validate','--write-sidecars'])
        log+=run(['node',args.drawio_cli,ROOT/(BASENAME+'.yaml'),tmp/'raw.svg','--validate'])
        (ROOT/'drawio-validation.log').write_text(log)
        svg=ET.parse(tmp/'raw.svg').getroot()
        svg.set('width',f'{width_pt:.8f}pt'); svg.set('height',f'{height_pt:.8f}pt')
        svg.set('viewBox',f'0 0 {canvas_w:g} {canvas_h:g}')
        svg.set('role','img')
        title=ET.Element(tag('title')); title.text='Study workflow: source-only class-definition optimization and frozen transfer evaluation'
        svg.insert(0,title)
        desc=ET.Element(tag('desc')); desc.text=spec['meta']['description']; svg.insert(1,desc)
        # A white background makes both the preview and publication export opaque.
        bg=ET.Element(tag('rect'),dict(x='0',y='0',width=f'{canvas_w:g}',height=f'{canvas_h:g}',fill='#FFFFFF'))
        svg.insert(2,bg)
        # The skill renderer uses a large marker; keep native open arrowheads,
        # with a restrained 7-canvas-unit footprint in final vector output.
        for marker in svg.iter(tag('marker')):
            marker.set('markerUnits','userSpaceOnUse'); marker.set('markerWidth','9'); marker.set('markerHeight','9')
        for parent in list(svg.iter()):
            for child in list(parent):
                if child.tag!=tag('text'): continue
                attrs=child.attrib; fs=float(attrs.get('font-size','24'))
                bold=int(attrs.get('font-weight','400'))>=600
                color=attrs.get('fill','#22323F')
                align=attrs.get('text-anchor','start')
                baseline=attrs.get('dominant-baseline','auto')
                y=float(attrs.get('y',0)); x=float(attrs.get('x',0))
                offset=fs*(.35 if baseline=='central' else .79 if baseline=='hanging' else 0)
                line_elements=list(child)
                lines=[]
                if line_elements:
                    for line in line_elements:
                        y+=float(line.get('dy','0'))
                        lines.append((''.join(line.itertext()),float(line.get('x',x)),y+offset))
                else: lines=[(''.join(child.itertext()),x,y+offset)]
                group=ET.Element(tag('g'),{'aria-label':' '.join(l[0] for l in lines)})
                for label,lx,ly in lines:
                    runs=parts(label,fs,bold)
                    total=sum(r[2] for r in runs)
                    cx=lx-total/2 if align=='middle' else lx-total if align=='end' else lx
                    for kind,value,advance,path in runs:
                        if kind=='text':
                            e=ET.SubElement(group,tag('text'),dict(x=f'{cx:.5f}',y=f'{ly:.5f}',fill=color,**{'font-family':'Arial','font-size':f'{fs:g}','font-weight':'700' if bold else '400','xml:space':'preserve'}))
                            e.text=value
                        else:
                            e=ET.SubElement(group,tag('path'),{'d':svg_path(path),'fill':color,'transform':f'translate({cx:.5f},{ly:.5f}) scale(1,-1)'})
                        cx+=advance
                idx=list(parent).index(child); parent.remove(child); parent.insert(idx,group)
        # Embed subset Arial fonts for portable SVG display as well as PDF.
        from fontTools import subset
        from fontTools.ttLib import TTFont
        css=[]
        for weight,key in [('400','regular'),('700','bold')]:
            chars=''.join(''.join(e.itertext()) for e in svg.iter(tag('text')) if e.get('font-weight','400')==weight)
            font=TTFont(font_files[key])
            options=subset.Options()
            options.recalc_timestamp=False
            sub=subset.Subsetter(options=options); sub.populate(text=chars); sub.subset(font)
            buf=io.BytesIO(); font.save(buf)
            encoded=base64.b64encode(buf.getvalue()).decode('ascii')
            css.append("@font-face {font-family: 'Arial'; font-style: normal; font-weight: "+weight+"; src: url(data:font/ttf;base64,"+encoded+") format('truetype');}")
        style=ET.Element(tag('style'),{'type':'text/css'}); style.text='\n'.join(css); svg.insert(2,style)
        svg_bytes=ET.tostring(svg,encoding='utf-8',xml_declaration=True)
        (ROOT/(BASENAME+'.svg')).write_bytes(svg_bytes)
        cairosvg.svg2pdf(bytestring=svg_bytes,write_to=str(ROOT/(BASENAME+'.pdf')))
        cairosvg.svg2png(bytestring=svg_bytes,write_to=str(ROOT/(BASENAME+'.png')),output_width=2000,output_height=round(2000*canvas_h/canvas_w))
        # Independent PDF rasterizer confirms that the exported PDF itself works.
        pdf=pymupdf.open(ROOT/(BASENAME+'.pdf'))
        page=pdf[0]
        page.get_pixmap(dpi=300,alpha=False).save(ROOT/(BASENAME+'-pdf-preview.png'))
        page.get_pixmap(dpi=200,colorspace=pymupdf.csGRAY,alpha=False).save(ROOT/(BASENAME+'-grayscale.png'))
        pdf_fonts=[]
        for f in page.get_fonts(full=True):
            name,ext,kind,data=pdf.extract_font(f[0])
            pdf_fonts.append(dict(name=name,extension=ext,type=kind,embedded_bytes=len(data)))
        raster_images=page.get_images(full=True)
        checks=dict(physical_width_pt=page.rect.width,physical_height_pt=page.rect.height,
            physical_width_mm=page.rect.width*25.4/72,physical_height_mm=page.rect.height*25.4/72,
            body_font_range_pt=[min(x['font_pt'] for x in label_metrics),max(x['font_pt'] for x in label_metrics)],
            body_font_range_pt_at_119mm=[min(x['font_px'] for x in label_metrics)*119/25.4*72/canvas_w,max(x['font_px'] for x in label_metrics)*119/25.4*72/canvas_w],
            min_line_width_pt=1.6*width_pt/canvas_w,fonts=pdf_fonts,
            pdf_raster_image_count=len(raster_images),svg_embedded_arial_subsets=2,labels=label_metrics,
            note='The skill heuristic counts text layers and intentional containing panels as nodes. Final acceptance uses measured Arial widths, native-vector inspection, and visual PDF/SVG/grayscale review.')
        (ROOT/'export-checks.json').write_text(json.dumps(checks,ensure_ascii=False,indent=2)+'\n')
        versions={p:importlib.metadata.version(p) for p in ['CairoSVG','matplotlib','PyMuPDF','PyYAML','fonttools']}
        versions.update(node=run(['node','--version']).strip(),font_files={k:str(v) for k,v in font_files.items()},font_sha256={k:hashlib.sha256(v.read_bytes()).hexdigest() for k,v in font_files.items()},drawio_cli=str(args.drawio_cli))
        (ROOT/'environment.json').write_text(json.dumps(versions,indent=2)+'\n')
        bad=[x for x in label_metrics if not x['horizontal_fits']]
        print(json.dumps(dict(dimensions_mm=[checks['physical_width_mm'],checks['physical_height_mm']],font_range_pt=checks['body_font_range_pt'],fonts=pdf_fonts,raster_images=len(raster_images),width_overflows=bad),ensure_ascii=False,indent=2))

if __name__=='__main__': main()
