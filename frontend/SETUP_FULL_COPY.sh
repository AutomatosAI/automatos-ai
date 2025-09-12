
#!/bin/bash

echo "🚀 AutomatOS AI Frontend - FULL COPY Setup"
echo "=========================================="
echo ""

# Check if we're in the right directory
if [ ! -f "package.json" ]; then
    echo "❌ Please run this script from the FULL_COPY directory"
    echo "   cd /home/ubuntu/automatos_ai_frontend/FULL_COPY"
    echo "   ./SETUP_FULL_COPY.sh"
    exit 1
fi

echo "📦 Installing dependencies..."
yarn install

echo ""
echo "🧹 Cleaning Next.js cache..."
rm -rf .next

echo ""
echo "✅ Setup complete!"
echo ""
echo "🚀 To start your frontend:"
echo "   yarn dev"
echo ""
echo "🔗 Then open: http://localhost:3000"
echo "🔑 Login with: admin@automatos.ai / admin"
echo ""
echo "📚 See README_FULL_COPY.md for detailed instructions"
echo ""

