

model_list = {  "RESNET" : "models.resnet18(pretrained=pretrained)",
                "GOOGLENET" :" models.googlenet(pretrained=pretrained)",
                "MOBILENET" : "models.mobilenet_v3_small(pretrained=pretrained)"}

for i,y in model_list.items():
    print(i + "---------" +y)