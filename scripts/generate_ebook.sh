#!/bin/bash
# Generate Automatos AI Context Engineering Ebook
# Converts markdown to PDF/EPUB with diagrams

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
DOCS_DIR="$PROJECT_ROOT/docs"
EBOOK_DIR="$PROJECT_ROOT/ebook"
EBOOK_SOURCE="$DOCS_DIR/EBOOK_CONTEXT_ENGINEERING.md"

echo "📚 Generating Automatos AI Context Engineering Ebook..."

# Create ebook directory
mkdir -p "$EBOOK_DIR"
cd "$EBOOK_DIR"

# Check dependencies
command -v pandoc >/dev/null 2>&1 || { echo "❌ pandoc is required. Install: brew install pandoc"; exit 1; }
command -v mermaid-cli >/dev/null 2>&1 || { echo "⚠️  mermaid-cli not found. Install: npm install -g @mermaid-js/mermaid-cli"; }

# Generate diagrams from Mermaid code
if command -v mmdc >/dev/null 2>&1; then
    echo "🎨 Generating diagrams..."
    mkdir -p "$EBOOK_DIR/diagrams"
    
    # Extract and render Mermaid diagrams
    # This is a simplified version - you may need to extract diagrams manually
    echo "   (Diagrams will be generated from Mermaid code in the markdown)"
else
    echo "⚠️  Skipping diagram generation (mermaid-cli not installed)"
fi

# Convert markdown to PDF
echo "📄 Generating PDF..."
pandoc "$EBOOK_SOURCE" \
    --from markdown \
    --to pdf \
    --output "$EBOOK_DIR/Automatos-AI-Context-Engineering.pdf" \
    --pdf-engine=xelatex \
    --toc \
    --toc-depth=3 \
    --number-sections \
    --highlight-style=tango \
    --geometry=margin=1in \
    --variable=fontsize:11pt \
    --variable=documentclass:book \
    --variable=mainfont:"Helvetica Neue" \
    --variable=monofont:"Menlo" \
    --variable=colorlinks:true \
    --variable=linkcolor:blue \
    --variable=citecolor:blue \
    --variable=urlcolor:blue \
    --metadata=title:"Context Engineering: The Complete Guide" \
    --metadata=author:"Automatos AI Platform" \
    --metadata=date:"$(date +%Y-%m-%d)" \
    || echo "⚠️  PDF generation failed. Install xelatex: brew install --cask mactex"

# Convert markdown to EPUB
echo "📱 Generating EPUB..."
pandoc "$EBOOK_SOURCE" \
    --from markdown \
    --to epub3 \
    --output "$EBOOK_DIR/Automatos-AI-Context-Engineering.epub" \
    --toc \
    --toc-depth=3 \
    --number-sections \
    --epub-cover-image="$PROJECT_ROOT/docs/assets/cover.png" \
    --metadata=title:"Context Engineering: The Complete Guide" \
    --metadata=author:"Automatos AI Platform" \
    --metadata=date:"$(date +%Y-%m-%d)" \
    || echo "⚠️  EPUB generation failed"

# Convert markdown to HTML (for web viewing)
echo "🌐 Generating HTML..."
pandoc "$EBOOK_SOURCE" \
    --from markdown \
    --to html5 \
    --output "$EBOOK_DIR/Automatos-AI-Context-Engineering.html" \
    --standalone \
    --toc \
    --toc-depth=3 \
    --number-sections \
    --css="$PROJECT_ROOT/docs/assets/ebook-style.css" \
    --metadata=title:"Context Engineering: The Complete Guide" \
    --metadata=author:"Automatos AI Platform" \
    || echo "⚠️  HTML generation failed"

echo ""
echo "✅ Ebook generation complete!"
echo "📁 Output directory: $EBOOK_DIR"
echo ""
echo "Generated files:"
ls -lh "$EBOOK_DIR"/*.{pdf,epub,html} 2>/dev/null || echo "   (Some formats may not have been generated)"
echo ""
echo "💡 Tips:"
echo "   • Install pandoc: brew install pandoc"
echo "   • Install xelatex for PDF: brew install --cask mactex"
echo "   • Install mermaid-cli for diagrams: npm install -g @mermaid-js/mermaid-cli"
echo "   • View HTML: open $EBOOK_DIR/Automatos-AI-Context-Engineering.html"

