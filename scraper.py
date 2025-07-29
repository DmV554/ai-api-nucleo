#!/usr/bin/env python3
"""
Web Scraper Completo para RAG
Navega automáticamente por sitios web, extrae contenido relevante
y genera datasets listos para procesos RAG.
"""

import requests
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException
from urllib.parse import urljoin, urlparse, parse_qs
import pandas as pd
import json
import time
import re
import logging
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import List, Dict, Set, Optional
import hashlib
from pathlib import Path

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class ScrapedContent:
    """Estructura para almacenar contenido extraído"""
    url: str
    title: str
    content: str
    metadata: Dict
    timestamp: str
    content_hash: str

class WebScraperRAG:
    def __init__(self, base_url: str, max_depth: int = 3, delay: float = 1.0):
        """
        Inicializa el scraper
        
        Args:
            base_url: URL base del sitio a scrapear
            max_depth: Profundidad máxima de navegación
            delay: Delay entre requests (respeto por el servidor)
        """
        self.base_url = base_url
        self.domain = urlparse(base_url).netloc
        self.max_depth = max_depth
        self.delay = delay
        self.visited_urls: Set[str] = set()
        self.scraped_data: List[ScrapedContent] = []
        
        # Configurar Selenium
        self.setup_selenium()
        
        # Patrones para identificar contenido relevante
        self.content_selectors = [
            'article', 'main', '.content', '#content', '.post', '.article',
            'section', '.text', '.description', 'p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6'
        ]
        
        # Elementos a ignorar
        self.ignore_selectors = [
            'nav', 'header', 'footer', '.nav', '.navigation', '.menu',
            '.sidebar', '.ads', '.advertisement', '.cookie', 'script', 'style'
        ]
        
        # Patrones de URLs a evitar
        self.url_patterns_to_avoid = [
            r'\.pdf$', r'\.jpg$', r'\.png$', r'\.gif$', r'\.zip$',
            r'/download/', r'/api/', r'/ajax/', r'#', r'javascript:',
            r'mailto:', r'tel:'
        ]

    def setup_selenium(self):
        """Configura el driver de Selenium"""
        options = webdriver.ChromeOptions()
        options.add_argument('--headless')
        options.add_argument('--no-sandbox')
        options.add_argument('--disable-dev-shm-usage')
        options.add_argument('--disable-gpu')
        options.add_argument('--window-size=1920,1080')
        options.add_argument('--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36')
        
        try:
            self.driver = webdriver.Chrome(options=options)
            self.driver.set_page_load_timeout(30)
        except Exception as e:
            logger.error(f"Error configurando Selenium: {e}")
            self.driver = None

    def is_valid_url(self, url: str) -> bool:
        """Verifica si una URL es válida para scrapear"""
        if not url or url in self.visited_urls:
            return False
            
        # Verificar patrones a evitar
        for pattern in self.url_patterns_to_avoid:
            if re.search(pattern, url, re.IGNORECASE):
                return False
        
        # Solo URLs del mismo dominio
        parsed = urlparse(url)
        return parsed.netloc == self.domain or parsed.netloc == ''

    def extract_links(self, soup: BeautifulSoup, current_url: str) -> List[str]:
        """Extrae todos los enlaces válidos de una página"""
        links = []
        
        # Enlaces en <a> tags
        for link in soup.find_all('a', href=True):
            url = urljoin(current_url, link['href'])
            if self.is_valid_url(url):
                links.append(url)
        
        # Botones con onclick o data attributes que puedan contener URLs
        for button in soup.find_all(['button', 'div'], {'onclick': True}):
            onclick = button.get('onclick', '')
            url_match = re.search(r"location\.href\s*=\s*['\"]([^'\"]+)['\"]", onclick)
            if url_match:
                url = urljoin(current_url, url_match.group(1))
                if self.is_valid_url(url):
                    links.append(url)
        
        return list(set(links))  # Remover duplicados

    def extract_content(self, soup: BeautifulSoup, url: str) -> ScrapedContent:
        """Extrae el contenido relevante de una página"""
        
        # Remover elementos no deseados
        for selector in self.ignore_selectors:
            for element in soup.select(selector):
                element.decompose()
        
        # Extraer título
        title = ""
        title_tag = soup.find('title')
        if title_tag:
            title = title_tag.get_text().strip()
        else:
            h1 = soup.find('h1')
            if h1:
                title = h1.get_text().strip()
        
        # Extraer contenido principal
        content_parts = []
        
        # Buscar contenido por selectores prioritarios
        for selector in self.content_selectors:
            elements = soup.select(selector)
            for element in elements:
                text = element.get_text().strip()
                if len(text) > 50:  # Solo texto significativo
                    content_parts.append(text)
        
        # Si no encontramos contenido, usar todo el texto del body
        if not content_parts:
            body = soup.find('body')
            if body:
                content_parts.append(body.get_text())
        
        # Limpiar y combinar contenido
        content = self.clean_text('\n\n'.join(content_parts))
        
        # Metadata adicional
        metadata = {
            'url': url,
            'domain': self.domain,
            'word_count': len(content.split()),
            'char_count': len(content),
            'title_length': len(title),
            'extraction_method': 'beautifulsoup'
        }
        
        # Agregar metadata de OpenGraph si existe
        og_title = soup.find('meta', property='og:title')
        og_description = soup.find('meta', property='og:description')
        if og_title:
            metadata['og_title'] = og_title.get('content', '')
        if og_description:
            metadata['og_description'] = og_description.get('content', '')
        
        # Hash del contenido para evitar duplicados
        content_hash = hashlib.md5(content.encode()).hexdigest()
        
        return ScrapedContent(
            url=url,
            title=title,
            content=content,
            metadata=metadata,
            timestamp=datetime.now().isoformat(),
            content_hash=content_hash
        )

    def clean_text(self, text: str) -> str:
        """Limpia y normaliza el texto extraído"""
        # Remover espacios extra y saltos de línea múltiples
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'\n\s*\n', '\n\n', text)
        
        # Remover caracteres especiales problemáticos
        text = re.sub(r'[^\w\s\.\,\!\?\;\:\-\(\)\[\]\{\}\"\'\/\@\#\$\%\&\*\+\=\<\>\|\~\`\^\n]', '', text)
        
        return text.strip()

    def scrape_with_requests(self, url: str) -> Optional[ScrapedContent]:
        """Scrapea una página usando requests (más rápido)"""
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            return self.extract_content(soup, url)
            
        except Exception as e:
            logger.warning(f"Error con requests en {url}: {e}")
            return None

    def scrape_with_selenium(self, url: str) -> Optional[ScrapedContent]:
        """Scrapea una página usando Selenium (para contenido dinámico)"""
        if not self.driver:
            return None
            
        try:
            self.driver.get(url)
            
            # Esperar a que cargue el contenido principal
            WebDriverWait(self.driver, 10).until(
                EC.presence_of_element_located((By.TAG_NAME, "body"))
            )
            
            # Hacer scroll para cargar contenido lazy
            self.driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(2)
            
            # Buscar y hacer clic en botones "Leer más", "Ver más", etc.
            expand_buttons = self.driver.find_elements(By.XPATH, 
                "//button[contains(text(), 'Leer más') or contains(text(), 'Ver más') or contains(text(), 'Mostrar más')]")
            
            for button in expand_buttons[:3]:  # Máximo 3 botones
                try:
                    self.driver.execute_script("arguments[0].click();", button)
                    time.sleep(1)
                except:
                    pass
            
            soup = BeautifulSoup(self.driver.page_source, 'html.parser')
            content = self.extract_content(soup, url)
            content.metadata['extraction_method'] = 'selenium'
            
            return content
            
        except Exception as e:
            logger.warning(f"Error con Selenium en {url}: {e}")
            return None

    def scrape_page(self, url: str) -> Optional[ScrapedContent]:
        """Scrapea una página probando diferentes métodos"""
        logger.info(f"Scrapeando: {url}")
        
        # Intentar primero con requests
        content = self.scrape_with_requests(url)
        
        # Si falla o el contenido es muy poco, usar Selenium
        if not content or len(content.content) < 200:
            content = self.scrape_with_selenium(url)
        
        if content and len(content.content) > 50:
            return content
        
        return None

    def crawl(self, start_url: str = None, depth: int = 0) -> None:
        """Crawlea el sitio web recursivamente"""
        if start_url is None:
            start_url = self.base_url
            
        if depth > self.max_depth or start_url in self.visited_urls:
            return
        
        self.visited_urls.add(start_url)
        
        # Scrapear la página actual
        content = self.scrape_page(start_url)
        if content:
            # Verificar duplicados por hash
            existing_hashes = {item.content_hash for item in self.scraped_data}
            if content.content_hash not in existing_hashes:
                self.scraped_data.append(content)
                logger.info(f"Contenido extraído: {len(content.content)} caracteres")
        
        # Encontrar enlaces para continuar el crawling
        if depth < self.max_depth:
            try:
                response = requests.get(start_url, timeout=10)
                soup = BeautifulSoup(response.content, 'html.parser')
                links = self.extract_links(soup, start_url)
                
                for link in links[:10]:  # Limitar enlaces por página
                    if link not in self.visited_urls:
                        time.sleep(self.delay)
                        self.crawl(link, depth + 1)
                        
            except Exception as e:
                logger.error(f"Error obteniendo enlaces de {start_url}: {e}")

    def save_dataset(self, output_path: str = None) -> str:
        """Guarda el dataset en múltiples formatos"""
        if not self.scraped_data:
            logger.warning("No hay datos para guardar")
            return ""
        
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            domain_clean = re.sub(r'[^\w\-_]', '_', self.domain)
            output_path = f"scraped_data_{domain_clean}_{timestamp}"
        
        output_dir = Path(output_path)
        output_dir.mkdir(exist_ok=True)
        
        # Convertir a lista de diccionarios
        data_dicts = [asdict(item) for item in self.scraped_data]
        
        # Guardar en JSON (formato completo)
        json_path = output_dir / "dataset.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(data_dicts, f, ensure_ascii=False, indent=2)
        
        # Guardar en CSV (versión simplificada)
        df = pd.DataFrame(data_dicts)
        csv_path = output_dir / "dataset.csv"
        df.to_csv(csv_path, index=False, encoding='utf-8')
        
        # Guardar en formato JSONL (ideal para RAG)
        jsonl_path = output_dir / "dataset.jsonl"
        with open(jsonl_path, 'w', encoding='utf-8') as f:
            for item in data_dicts:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        
        # Guardar solo texto plano (para algunos sistemas RAG)
        txt_path = output_dir / "corpus.txt"
        with open(txt_path, 'w', encoding='utf-8') as f:
            for item in self.scraped_data:
                f.write(f"=== {item.title} ===\n")
                f.write(f"URL: {item.url}\n")
                f.write(f"{item.content}\n\n")
        
        # Estadísticas
        stats = {
            'total_pages': len(self.scraped_data),
            'total_words': sum(len(item.content.split()) for item in self.scraped_data),
            'total_chars': sum(len(item.content) for item in self.scraped_data),
            'average_words_per_page': sum(len(item.content.split()) for item in self.scraped_data) / len(self.scraped_data),
            'urls_visited': len(self.visited_urls),
            'domain': self.domain,
            'timestamp': datetime.now().isoformat()
        }
        
        stats_path = output_dir / "stats.json"
        with open(stats_path, 'w', encoding='utf-8') as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Dataset guardado en: {output_dir}")
        logger.info(f"Páginas procesadas: {stats['total_pages']}")
        logger.info(f"Palabras totales: {stats['total_words']}")
        
        return str(output_dir)

    def close(self):
        """Cierra recursos"""
        if self.driver:
            self.driver.quit()

def main():
    """Función principal para ejecutar el scraper"""
    
    # Configuración
    BASE_URL = "https://valparaisoweb.cl/"  # Cambiar por la URL objetivo
    MAX_DEPTH = 2
    DELAY = 1.0
    
    # Crear y ejecutar scraper
    scraper = WebScraperRAG(BASE_URL, max_depth=MAX_DEPTH, delay=DELAY)
    
    try:
        logger.info(f"Iniciando scraping de: {BASE_URL}")
        scraper.crawl()
        
        # Guardar resultados
        output_path = scraper.save_dataset("data/scraped_data")
        
        print(f"\n✅ Scraping completado!")
        print(f"📁 Datos guardados en: {output_path}")
        print(f"📄 Páginas procesadas: {len(scraper.scraped_data)}")
        
    except KeyboardInterrupt:
        logger.info("Scraping interrumpido por el usuario")
    except Exception as e:
        logger.error(f"Error durante el scraping: {e}")
    finally:
        scraper.close()

if __name__ == "__main__":
    main()

# Ejemplo de uso avanzado:
"""
# Para usar el scraper programáticamente:

scraper = WebScraperRAG("https://valparaisoweb.cl/", max_depth=3, delay=0.5)
scraper.crawl()

# Filtrar contenido por longitud mínima
scraper.scraped_data = [
    item for item in scraper.scraped_data 
    if len(item.content.split()) > 100
]

# Guardar con nombre personalizado
output_path = scraper.save_dataset("mi_dataset_rag")
scraper.close()
"""