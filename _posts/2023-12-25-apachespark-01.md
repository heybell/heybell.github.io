---
title: Apache Spark 독립 실행형 클러스터 (1)
date: 2023-12-25 22:30:00 +09:00
categories: [Data Engineer, Apache Spark]
tags: [Data Enginner, Ubuntu 22.04, Standalone Cluster, Apache Spark]
---

Apache Spark는 대규모 데이터 처리와 분석에 효과적인 오픈 소스 분산 컴퓨팅 시스템으로 널리 사용되고 있다. 개인 프로젝트의 규모를 고려하면 더 가벼운 도구를 검토하는 것이 합리적일 수 있지만, Spark를 선택한 이유는 클러스터를 활용한 병렬 처리에 대한 이해를 높이고자 하기 때문이다. 또한, PySpark, SparkR, Scala 등의 API를 사용하여 사용자가 선호하는 언어로 ETL 작업을 구현할 수 있기 때문에 채택하였다.

그 중 본 포스팅은 독립 실행형 클러스터(Standalone Cluster)에 대해 다루려고 한다. 독립 실행형 클러스터는 별도의 외부 클러스터 매니저(예: Apache Mesos, Apache Hadoop YARN, Kubernetes)를 사용하지 않고 Spark가 제공하는 내장된 클러스터 매니저를 사용하여 master와 worker 노드를 설정하고 실행한다.

## **운영 환경**
- 플랫폼 : Proxmox
- 운영체제 : Ubuntu 22.04
- JDK : 1.8
- Scala : 2.11.12
- Python : 3.10.12

## **Java JDK 설치**
각 가상머신, master 및 worker에서 다음 명령어들을 따르고 Java JDK를 설치한다.

```shell
sudo apt-get install software_properties_common
sudo add-apt-repository ppa:webupd8team/java
sudo apt-get update
sudo apt-get install openjdk-8-jdk
```

## **Scala 설치**
각 가상머신, master 및 worker에서 다음 명령어를 사용하여 Scala를 설치한다.

```shell
sudo apt-get install scala
```

## **호스트 이름 변경**
각 가상머신, master 및 worker에서 다음 명령어를 사용하여 호스트 이름을 입력한다.

```shell
sudo nano /hostname
```

![Desktop View](/assets/img/screenshot/스크린샷 2023-12-25 024838.png)
![Desktop View](/assets/img/screenshot/스크린샷 2023-12-25 025134.png)
![Desktop View](/assets/img/screenshot/스크린샷 2023-12-25 025108.png)

## **IP 주소 확인**
다음 명령어를 입력하여 IP 주소를 확인한다.

```shell
ip addr
```

- spark-master : 10.0.0.210
- spark-worker1 : 10.0.0.211
- spark-worker2 : 10.0.0.212

## **호스트 파일 편집**
각 가상머신, master 및 worker에서 다음 명령어를 사용하여 네트워크 정보를 추가한다.

```shell
sudo nano /etc/hosts
```

![Desktop View](/assets/img/screenshot/스크린샷 2023-12-25 204521.png){: w="500" h="400" }
![Desktop View](/assets/img/screenshot/스크린샷 2023-12-25 204544.png){: w="500" h="400" }
![Desktop View](/assets/img/screenshot/스크린샷 2023-12-25 204605.png){: w="500" h="400" }

## **가상머신 재부팅**
각 가상머신, master 및 worker에서 다음 명령어를 사용하여 재부팅한다.

```shell
sudo reboot
```

## **SSH 구성**
이 단계는 master에서만 수행되며, 다음 명령어들을 따른다.

- Open SSH Server-Client를 설치

```shell
sudo apt-get install openssh-server openssh-client
```

- SSH 키 쌍 생성

```shell
ssh-keygen -t rsa -P ""
```

- SSH 공개 키를 authorized_keys 파일에 추가

```shell
cat ~/.ssh/id_rsa.pub >> ~/.ssh/authorized_keys
```

- .ssh/id_rsa.pub(master)의 내용을 .ssh/authorized_keys (master 및 worker)에 복사

```shell
ssh-copy-id spark@spark-master
ssh-copy-id spark@spark-worker1
ssh-copy-id spark@spark-worker2
```

- 위 과정들이 잘 되었는지 확인하기 위해 worker에 연결

```shell
ssh spark-worker1
ssh spark-worker2
```

## **Apache Spark 설치 및 설정**
각 가상머신, master 및 worker에서 다음 명령어들을 따른다.

- Apache Spark 다운로드

```shell
wget https://archive.apache.org/dist/spark/spark-2.4.8/spark-2.4.8-bin-hadoop2.7.tgz
```

- Apache Spark 압축 풀기

```shell
tar xvf spark-2.4.8-bin-hadoop2.7.tgz
```

- Spark 소프트웨어 파일을 해당 디렉터리(/usr/local/bin)로 이동

```shell
sudo mv spark-2.4.8-bin-hadoop2.7 /usr/local/spark
```

- Apache Spark 환경 설정

```shell
sudo nano ~/.bashrc
```

```shell
# 맨 아래에 추가
export PATH=$PATH:/usr/local/spark/bin
```

## **Apache Spark master 구성**
이 단계는 master에서만 수행되며, 다음 명령어들을 따른다.

- 해당 디렉터리(/usr/local/spark/conf)로 이동하여 spark-env.sh 템플릿의 복사본을 만들고 이름 변경

```shell
cd /usr/local/spark/conf
cp spark-env.sh.template spark-env.sh
```

- spark-env.sh 편집

```shell
sudo vim spark-env.sh
```

```shell
# 맨 아래에 추가
export SPARK_MASTER_HOST='10.0.0.210'
export JAVA_HOME=/usr/lib/jvm/java-8-openjdk-amd64
export PYSPARK_PYTHON=python3
```

- worker 추가

```shell
sudo nano slaves
```

```shell
# worker 호스트명 또는 IP 추가
spark-worker1
spark-worker2
```

## **Apache Spark 클러스터 실행 및 중지**
master에서 다음 명령어들을 실행하면 된다.

- Apache Spark 클러스터 실행

```shell
cd /usr/local/spark
./sbin/start-all.sh
```

- Apache Spark 클러스터 중지

```shell
cd /usr/local/spark
./sbin/stop-all.sh
```

- 서비스 시작 확인

```shell
cd /usr/local/spark
jps
```

- Spark UI를 탐색하여 클러스터에 대해 알아보기 : <http://10.0.0.210:8080/>
![Desktop View](/assets/img/screenshot/스크린샷 2023-12-25 220306.png)