FROM node:18-alpine

# Set working directory
WORKDIR /app

# Copy package files
COPY package.json yarn.lock* package-lock.json* ./

# Install dependencies
RUN npm ci --legacy-peer-deps

# Copy source code
COPY . .

# Create environment file
RUN touch .env.local

# Build the application
RUN npm run build

# Remove dev dependencies after build
RUN npm prune --production

# Start the application
CMD [npm, start]
