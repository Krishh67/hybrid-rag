import threading
import queue
import time
import logging

logger = logging.getLogger(__name__)

class RAGWorker:
    def __init__(self):
        self.request_queue = queue.Queue()
        self.response_queue = queue.Queue()
        self.progress_queue = queue.Queue()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self):
        with open("rag_worker.log", "a") as f:
            f.write("TRACE: Worker thread starting\n")
            
        # We import everything INSIDE the persistent worker thread
        import os
        os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
        
        with open("rag_worker.log", "a") as f:
            f.write("TRACE: Importing RetrievalService\n")
        
        try:
            from services.retrieval_service import RetrievalService
            with open("rag_worker.log", "a") as f:
                f.write("TRACE: Successfully imported RetrievalService\n")
        except Exception as e:
            with open("rag_worker.log", "a") as f:
                f.write(f"TRACE: Error importing RetrievalService: {e}\n")
            raise
        
        logger.info("RAG Worker thread started.")
        while True:
            try:
                task = self.request_queue.get()
                if task is None:
                    break
                    
                with open("rag_worker.log", "a") as f:
                    f.write("TRACE: Received task\n")
                    
                kb_dir = task["kb_dir"]
                query = task["query"]
                
                # Callback to send progress back to Streamlit
                def progress_callback(msg):
                    self.progress_queue.put(msg)
                    
                with open("rag_worker.log", "a") as f:
                    f.write("TRACE: About to instantiate RetrievalService\n")
                service = RetrievalService(kb_dir)
                
                with open("rag_worker.log", "a") as f:
                    f.write("TRACE: About to execute_query\n")
                trace = service.execute_query(query, progress_callback=progress_callback)
                
                with open("rag_worker.log", "a") as f:
                    f.write("TRACE: Finished execute_query\n")
                self.response_queue.put({"status": "success", "trace": trace})
                
            except Exception as e:
                logger.error("Error in RAGWorker: %s", e, exc_info=True)
                self.response_queue.put({"status": "error", "error": str(e)})
                
    def query(self, kb_dir, query_text):
        # Clear queues
        while not self.response_queue.empty(): self.response_queue.get()
        while not self.progress_queue.empty(): self.progress_queue.get()
        
        self.request_queue.put({"kb_dir": kb_dir, "query": query_text})
        
    def get_progress(self):
        updates = []
        while not self.progress_queue.empty():
            updates.append(self.progress_queue.get())
        return updates
        
    def get_result(self, timeout=0.1):
        try:
            return self.response_queue.get(timeout=timeout)
        except queue.Empty:
            return None
