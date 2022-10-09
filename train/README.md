# Run
```python3
python3 numerical_flow.py --data_name=longlat-200M-100R-zipf --plot=True \
                          --encoder_type=partition --shifts=1000000 \
                          --decoder_type=sum --num_flows=1 --num_layers=2 \
                          --input_dim=2 --hidden_dim=1 --steps=500 \
                          --train_ratio=0.1
```