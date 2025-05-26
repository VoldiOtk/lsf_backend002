FROM nginx:alpine

# Copier la configuration
COPY nginx.conf /etc/nginx/nginx.conf

# Créer le répertoire pour les certificats SSL
RUN mkdir -p /etc/nginx/ssl

# Créer le répertoire pour les logs
RUN mkdir -p /var/log/nginx

# Exposer les ports HTTP et HTTPS
EXPOSE 80 443 