import hashlib
import json
import time
import datetime

class BlockchainLogger:
    """
    Simple blockchain implementation for log integrity verification
    """
    
    def __init__(self):
        self.chain = []
        # Create the genesis block
        self.create_genesis_block()
    
    def create_genesis_block(self):
        """
        Create the first block in the chain (genesis block)
        """
        genesis_block = {
            'index': 0,
            'timestamp': str(datetime.datetime.now()),
            'data': "Genesis Block",
            'previous_hash': "0",
            'hash': self.calculate_hash(0, "0", "Genesis Block", str(datetime.datetime.now()))
        }
        self.chain.append(genesis_block)
    
    def get_latest_block(self):
        """
        Return the most recent block in the chain
        """
        return self.chain[-1]
    
    def calculate_hash(self, index, previous_hash, data, timestamp):
        """
        Calculate the hash of a block
        """
        # Make data JSON serializable by converting non-serializable types
        if isinstance(data, dict):
            serializable_data = {}
            for key, value in data.items():
                # Convert non-serializable types to strings
                if hasattr(value, 'isoformat') and callable(getattr(value, 'isoformat')):
                    # Handle datetime and Timestamp objects
                    serializable_data[key] = str(value)
                elif str(type(value)).find('numpy') >= 0:
                    # Handle numpy types
                    serializable_data[key] = str(value)
                else:
                    serializable_data[key] = value
            data = serializable_data
            
        # Convert to JSON string
        if isinstance(data, dict) or isinstance(data, list):
            try:
                data = json.dumps(data, sort_keys=True)
            except TypeError:
                # If still not serializable, convert to string
                data = str(data)
        
        # Create the block string to hash
        block_string = f"{index}{previous_hash}{data}{timestamp}"
        
        # Return the SHA-256 hash
        return hashlib.sha256(block_string.encode()).hexdigest()
    
    def add_log(self, log_data):
        """
        Add a new log to the blockchain
        """
        latest_block = self.get_latest_block()
        index = latest_block['index'] + 1
        timestamp = str(datetime.datetime.now())
        previous_hash = latest_block['hash']
        
        # Create new block
        block = {
            'index': index,
            'timestamp': timestamp,
            'data': log_data,
            'previous_hash': previous_hash,
            'hash': self.calculate_hash(index, previous_hash, log_data, timestamp)
        }
        
        self.chain.append(block)
        return block
    
    def is_chain_valid(self):
        """
        Check if the blockchain is valid (all hashes match and links are correct)
        """
        for i in range(1, len(self.chain)):
            current_block = self.chain[i]
            previous_block = self.chain[i-1]
            
            # Check if the current block's hash is correct
            if current_block['hash'] != self.calculate_hash(
                current_block['index'], 
                current_block['previous_hash'], 
                current_block['data'], 
                current_block['timestamp']
            ):
                return False
            
            # Check if the previous hash reference is correct
            if current_block['previous_hash'] != previous_block['hash']:
                return False
        
        return True
    
    def get_chain(self):
        """
        Return the entire blockchain
        """
        return self.chain
    
    def verify_block(self, block_index):
        """
        Verify a specific block in the chain
        """
        if block_index < 0 or block_index >= len(self.chain):
            return False
        
        block = self.chain[block_index]
        
        # Calculate the expected hash
        expected_hash = self.calculate_hash(
            block['index'], 
            block['previous_hash'], 
            block['data'], 
            block['timestamp']
        )
        
        # Check if the stored hash matches the calculated hash
        return block['hash'] == expected_hash
