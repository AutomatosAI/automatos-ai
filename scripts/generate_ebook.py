#!/usr/bin/env python3
"""
Generate Automatos AI Context Engineering Ebook
Converts markdown to PDF/EPUB/HTML with diagrams
"""

import os
import sys
import subprocess
from pathlib import Path
from datetime import datetime

def check_dependency(cmd, install_cmd=None):
    """Check if a command is available"""
    try:
        subprocess.run([cmd, "--version"], capture_output=True, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        if install_cmd:
            print(f"❌ {cmd} not found. Install: {install_cmd}")
        return False

def generate_diagrams(source_file, output_dir):
    """Extract and generate Mermaid diagrams"""
    diagrams_dir = output_dir / "diagrams"
    diagrams_dir.mkdir(exist_ok=True)
    
    print("🎨 Extracting Mermaid diagrams...")
    
    # Read source file
    with open(source_file, 'r') as f:
        content = f.read()
    
    # Find Mermaid code blocks
    import re
    mermaid_pattern = r'```mermaid\n(.*?)\n```'
    diagrams = re.findall(mermaid_pattern, content, re.DOTALL)
    
    if not diagrams:
        print("   No Mermaid diagrams found")
        return
    
    if not check_dependency("mmdc"):
        print("   ⚠️  mermaid-cli not installed. Skipping diagram generation.")
        print("   Install: npm install -g @mermaid-js/mermaid-cli")
        return
    
    print(f"   Found {len(diagrams)} diagram(s)")
    
    for i, diagram_code in enumerate(diagrams, 1):
        diagram_file = diagrams_dir / f"diagram_{i}.mmd"
        output_file = diagrams_dir / f"diagram_{i}.png"
        
        # Write diagram to temp file
        with open(diagram_file, 'w') as f:
            f.write(diagram_code)
        
        # Generate PNG
        try:
            subprocess.run(
                ["mmdc", "-i", str(diagram_file), "-o", str(output_file)],
                check=True,
                capture_output=True
            )
            print(f"   ✅ Generated: {output_file.name}")
        except subprocess.CalledProcessError as e:
            print(f"   ❌ Failed to generate diagram {i}: {e}")

def generate_pdf(source_file, output_file, metadata):
    """Generate PDF using pandoc"""
    if not check_dependency("pandoc"):
        print("❌ pandoc not found. Install: brew install pandoc")
        return False
    
    print("📄 Generating PDF...")
    
    cmd = [
        "pandoc",
        str(source_file),
        "--from", "markdown",
        "--to", "pdf",
        "--output", str(output_file),
        "--pdf-engine=xelatex",
        "--toc",
        "--toc-depth=3",
        "--number-sections",
        "--highlight-style=tango",
        "--geometry=margin=1in",
        "--variable=fontsize:11pt",
        "--variable=documentclass:book",
        "--variable=mainfont:Helvetica Neue",
        "--variable=monofont:Menlo",
        "--variable=colorlinks:true",
        "--metadata", f"title:{metadata['title']}",
        "--metadata", f"author:{metadata['author']}",
        "--metadata", f"date:{metadata['date']}",
    ]
    
    try:
        subprocess.run(cmd, check=True, capture_output=True)
        print(f"   ✅ Generated: {output_file.name}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"   ❌ PDF generation failed: {e.stderr.decode()}")
        print("   💡 Install xelatex: brew install --cask mactex")
        return False

def generate_epub(source_file, output_file, metadata, cover_image=None):
    """Generate EPUB using pandoc"""
    if not check_dependency("pandoc"):
        return False
    
    print("📱 Generating EPUB...")
    
    cmd = [
        "pandoc",
        str(source_file),
        "--from", "markdown",
        "--to", "epub3",
        "--output", str(output_file),
        "--toc",
        "--toc-depth=3",
        "--number-sections",
        "--metadata", f"title:{metadata['title']}",
        "--metadata", f"author:{metadata['author']}",
        "--metadata", f"date:{metadata['date']}",
    ]
    
    if cover_image and cover_image.exists():
        cmd.extend(["--epub-cover-image", str(cover_image)])
    
    try:
        subprocess.run(cmd, check=True, capture_output=True)
        print(f"   ✅ Generated: {output_file.name}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"   ❌ EPUB generation failed: {e.stderr.decode()}")
        return False

def generate_html(source_file, output_file, metadata, css_file=None):
    """Generate HTML using pandoc"""
    if not check_dependency("pandoc"):
        return False
    
    print("🌐 Generating HTML...")
    
    cmd = [
        "pandoc",
        str(source_file),
        "--from", "markdown",
        "--to", "html5",
        "--output", str(output_file),
        "--standalone",
        "--toc",
        "--toc-depth=3",
        "--number-sections",
        "--metadata", f"title:{metadata['title']}",
        "--metadata", f"author:{metadata['author']}",
        "--metadata", f"date:{metadata['date']}",
    ]
    
    if css_file and css_file.exists():
        cmd.extend(["--css", str(css_file)])
    
    try:
        subprocess.run(cmd, check=True, capture_output=True)
        print(f"   ✅ Generated: {output_file.name}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"   ❌ HTML generation failed: {e.stderr.decode()}")
        return False

def main():
    """Main function"""
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    docs_dir = project_root / "docs"
    ebook_dir = project_root / "ebook"
    assets_dir = docs_dir / "assets"
    
    source_file = docs_dir / "EBOOK_CONTEXT_ENGINEERING.md"
    
    if not source_file.exists():
        print(f"❌ Source file not found: {source_file}")
        sys.exit(1)
    
    # Create output directory
    ebook_dir.mkdir(exist_ok=True)
    
    print("📚 Generating Automatos AI Context Engineering Ebook...")
    print(f"   Source: {source_file}")
    print(f"   Output: {ebook_dir}\n")
    
    # Metadata
    metadata = {
        "title": "Context Engineering: The Complete Guide to Intelligent AI Orchestration",
        "author": "Automatos AI Platform",
        "date": datetime.now().strftime("%Y-%m-%d")
    }
    
    # Generate diagrams
    generate_diagrams(source_file, ebook_dir)
    print()
    
    # Generate formats
    base_name = "Automatos-AI-Context-Engineering"
    css_file = assets_dir / "ebook-style.css"
    cover_image = assets_dir / "cover.png"
    
    generate_pdf(source_file, ebook_dir / f"{base_name}.pdf", metadata)
    print()
    
    generate_epub(source_file, ebook_dir / f"{base_name}.epub", metadata, cover_image)
    print()
    
    generate_html(source_file, ebook_dir / f"{base_name}.html", metadata, css_file)
    print()
    
    # Summary
    print("✅ Ebook generation complete!")
    print(f"📁 Output directory: {ebook_dir}")
    print()
    print("Generated files:")
    for ext in ["pdf", "epub", "html"]:
        file = ebook_dir / f"{base_name}.{ext}"
        if file.exists():
            size = file.stat().st_size / 1024  # KB
            print(f"   • {file.name} ({size:.1f} KB)")
    print()
    print("💡 Tips:")
    print("   • View HTML: open ebook/Automatos-AI-Context-Engineering.html")
    print("   • Install pandoc: brew install pandoc")
    print("   • Install xelatex for PDF: brew install --cask mactex")
    print("   • Install mermaid-cli: npm install -g @mermaid-js/mermaid-cli")

if __name__ == "__main__":
    main()

