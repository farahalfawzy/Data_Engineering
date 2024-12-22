# import pandas as pd
# from kafka import KafkaConsumer
# import json
# from cleaning import clean  # Import your cleaning function if needed
# def start_consumer():
# # Initialize Kafka consumer
#     consumer = KafkaConsumer(
#         'fintech',  # Topic name
#         bootstrap_servers=['kafka:9092'],
#         auto_offset_reset='latest',
#         value_deserializer=lambda x: json.loads(x.decode('utf-8')),
#     )

#     print("Listening for messages in 'fintech'...")
#     messages = []
# #
#     column_names = [
#         "Customer Id", "Emp Title", "Emp Length", "Home Ownership", "Annual Inc", 
#         "Annual Inc Joint", "Verification Status", "Zip Code", "Addr State", 
#         "Avg Cur Bal", "Tot Cur Bal", "Loan Id", "Loan Status", "Loan Amount", 
#         "State", "Funded Amount", "Term", "Int Rate", "Grade", "Issue Date", 
#         "Pymnt Plan", "Type", "Purpose", "Description"
#     ]

#     while True:
#         message_batch = consumer.poll(timeout_ms=2000)
        
#         if message_batch:
#             for tp, messages_list in message_batch.items():
#                 for msg in messages_list:
#                     message_value = msg.value
#                     print(f"Received message: {message_value}")
                    
#                     # Check if message is EOF
#                     if message_value == "EOF":
#                         print("EOF message received. Stopping consumer.")
#                         consumer.close()
                        
#                         # Convert collected messages to DataFrame
#                         df = pd.DataFrame(messages, columns=column_names)
#                         print("All messages received:")
#                         print(df)
#                         return df  # Return DataFrame containing all messages

#                     # Collect the message if it's not EOF
#                     messages.append(message_value)
#         else:
#             print("No new messages, waiting for next poll...")
            

#     consumer.close()
#     print("Consumer closed.")

import pandas as pd
from kafka import KafkaConsumer
import json

def start_consumer():
    """
    Initializes the Kafka consumer and returns it.
    """
    consumer = KafkaConsumer(
        'fintech',  # Topic name
        bootstrap_servers=['kafka:9092'],
        auto_offset_reset='latest',
        enable_auto_commit=True,
        value_deserializer=lambda x: json.loads(x.decode('utf-8'))
    )
    print("Kafka consumer initialized.")
    return consumer


    
def get_next_message(consumer):
  
    message_batch = consumer.poll(timeout_ms=2000)

    if message_batch:
        for tp, messages_list in message_batch.items():
            for msg in messages_list:
                message_value = msg.value
                print(f"Received message: {message_value}")
                return message_value
              
    print("No new messages, waiting for next poll...")
    return None  # Return empty DataFrame if no new messages

def close_consumer(consumer):
    """
    Closes the Kafka consumer.
    """
    consumer.close()
    print("Kafka consumer closed.")
